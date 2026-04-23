import argparse
import asyncio
import traceback
from typing import Dict, List

import uvloop
import zmq
import zmq.asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from omniserve.engine.arg_utils import EngineArgs
from omniserve.engine.llm_engine import LLMEngine

from serving.io_struct import BatchTokenIDOut, TokenizedGenerateReqInput
from serving.log_utils import elog, ilog, vlog


class Router:
    def __init__(
        self,
        llm_engine: LLMEngine,
        router_port: int,
        detokenizer_port: int,
        forward_sleep: float = 0.001,
    ):
        ctx = zmq.asyncio.Context(2)
        self.recv_from_tokenizer = ctx.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{router_port}")
        self.send_to_detokenizer = ctx.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(f"tcp://127.0.0.1:{detokenizer_port}")

        self.llm_engine = llm_engine
        self.pending: List[TokenizedGenerateReqInput] = []
        self.forward_sleep = forward_sleep

        # Engine's output dicts key by internal integer seq_id, but the
        # TokenizerManager knows requests by the external string rid.
        # Translate between the two at admit / send time.
        self.seq_id_to_rid: Dict[int, str] = {}
        self.seq_id_to_input_len: Dict[int, int] = {}

    async def loop_for_recv_requests(self) -> None:
        while True:
            try:
                req: TokenizedGenerateReqInput = await self.recv_from_tokenizer.recv_pyobj()
            except Exception as e:
                elog(f"[router] recv error: {e!r}")
                await asyncio.sleep(0.1)
                continue
            vlog(f"[router] recv rid={req.rid} input_len={len(req.input_ids)}")
            self.pending.append(req)

    async def loop_for_forward(self) -> None:
        step_count = 0
        while True:
            if self.pending:
                new_reqs = self.pending
                self.pending = []
                vlog(f"[router] admitting {len(new_reqs)} new req(s)")
                await self._admit_new_requests(new_reqs)

            if self.llm_engine.has_unfinished_requests():
                try:
                    outs = self.llm_engine.step()
                except Exception as e:
                    traceback.print_exc()
                    elog(f"[router] engine.step failed: {e!r}")
                    outs = []
                step_count += 1
                if step_count <= 3 or step_count % 16 == 0:
                    finished_now = sum(1 for o in outs if o.get("finished"))
                    vlog(
                        f"[router] step#{step_count} outs={len(outs)} finished={finished_now}",
                    )
                if outs:
                    await self._send_step_outputs(outs)

            await asyncio.sleep(self.forward_sleep)

    async def _admit_new_requests(
        self, reqs: List[TokenizedGenerateReqInput]
    ) -> None:
        rejected_rids: List[str] = []
        for req in reqs:
            try:
                accepted = self.llm_engine.add_request(
                    request_id=req.rid,
                    prompt=None,
                    sampling_params=req.sampling_params,
                    prompt_token_ids=req.input_ids,
                )
            except Exception as e:
                traceback.print_exc()
                elog(f"[router] add_request failed for rid={req.rid}: {e!r}")
                accepted = False

            if accepted is False:
                rejected_rids.append(req.rid)
            else:
                # add_seq_group appends to scheduler.waiting; the just-added
                # group is at the tail. Pull out its seq_id so we can map
                # engine outputs (keyed by seq_id int) back to the external
                # string rid.
                seq_group = self.llm_engine.scheduler.waiting[-1]
                assert seq_group.request_id == req.rid, (
                    f"unexpected tail in scheduler.waiting: "
                    f"got {seq_group.request_id!r}, expected {req.rid!r}"
                )
                seq_id = seq_group.get_seqs()[0].seq_id
                self.seq_id_to_rid[seq_id] = req.rid
                self.seq_id_to_input_len[seq_id] = len(req.input_ids)

        if rejected_rids:
            await self.send_to_detokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids=rejected_rids,
                    output_tokens=[[] for _ in rejected_rids],
                    finished=[True for _ in rejected_rids],
                    final_text=[None for _ in rejected_rids],
                    errors=["prompt rejected by engine (too long?)" for _ in rejected_rids],
                )
            )

    async def _send_step_outputs(self, outs: List[dict]) -> None:
        rids: List[str] = []
        output_tokens: List[List[int]] = []
        finished: List[bool] = []
        final_text: List = []
        errors: List = []

        for o in outs:
            seq_id = o["id"]
            rid = self.seq_id_to_rid.get(seq_id)
            if rid is None:
                elog(f"[router] WARN: got output for unknown seq_id={seq_id}")
                continue

            is_finished = bool(o.get("finished", False))
            prompt_len = self.seq_id_to_input_len.get(seq_id, 0)

            # OmniServe's engine output is asymmetric:
            #   not-finished: {"tokens": [prompt+gen ids], no "text"}
            #   finished:     {"text":  <prompt+gen text>, no "tokens"}
            # So on finished frames we must use engine's text (and it
            # includes the prompt prefix). TokenizerManager strips the
            # prompt on its side using the original prompt string.
            if is_finished:
                rids.append(rid)
                output_tokens.append([])
                finished.append(True)
                final_text.append(o.get("text"))
                errors.append(None)
                self.seq_id_to_rid.pop(seq_id, None)
                self.seq_id_to_input_len.pop(seq_id, None)
            else:
                full_tokens = o.get("tokens") or []
                gen = full_tokens[prompt_len:]
                rids.append(rid)
                output_tokens.append(list(gen))
                finished.append(False)
                final_text.append(None)
                errors.append(None)

        vlog(f"[router] send BatchTokenIDOut rids={rids} finished={finished}")
        await self.send_to_detokenizer.send_pyobj(
            BatchTokenIDOut(
                rids=rids,
                output_tokens=output_tokens,
                finished=finished,
                final_text=final_text,
                errors=errors,
            )
        )


def start_router_process(
    engine_cli_args: List[str],
    router_port: int,
    detokenizer_port: int,
    pipe_writer,
) -> None:
    try:
        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)
        args = parser.parse_args(engine_cli_args)
        engine_args = EngineArgs.from_cli_args(args)

        ilog("[router] building engine (this may take a while)...")
        engine = LLMEngine.from_engine_args(engine_args)
        ilog("[router] engine ready")

        router = Router(
            engine,
            router_port=router_port,
            detokenizer_port=detokenizer_port,
        )
    except Exception:
        tb = traceback.format_exc()
        try:
            pipe_writer.send(("error", tb))
        finally:
            pipe_writer.close()
        raise

    pipe_writer.send(("ok", None))
    pipe_writer.close()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_recv_requests())
    loop.run_until_complete(router.loop_for_forward())

import asyncio
import traceback
from typing import Optional

import uvloop
import zmq
import zmq.asyncio

from omniserve.utils.tokenizer import get_tokenizer

from serving.io_struct import BatchStrOut, BatchTokenIDOut
from serving.log_utils import elog, ilog, vlog


class DetokenizerManager:
    def __init__(
        self,
        tokenizer_name: str,
        detokenizer_port: int,
        tokenizer_port: int,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
    ):
        self.tokenizer = get_tokenizer(
            tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )

        ctx = zmq.asyncio.Context(2)
        # router -> detokenizer
        self.recv_from_router = ctx.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{detokenizer_port}")
        # detokenizer -> tokenizer manager
        self.send_to_tokenizer = ctx.socket(zmq.PUSH)
        self.send_to_tokenizer.connect(f"tcp://127.0.0.1:{tokenizer_port}")

    async def handle_loop(self) -> None:
        ilog("[detok] handle_loop started")
        while True:
            try:
                obj: BatchTokenIDOut = await self.recv_from_router.recv_pyobj()
            except Exception as e:
                elog(f"[detok] recv error: {e!r}")
                await asyncio.sleep(0.1)
                continue

            n = len(obj.rids)
            output_strs = [""] * n
            errors = list(obj.errors) if obj.errors else [None] * n
            final_text = obj.final_text if obj.final_text else [None] * n

            for i in range(n):
                if errors[i] is not None:
                    continue

                if obj.finished[i] and final_text[i] is not None:
                    # Engine's authoritative final text (stop-string trimmed etc.).
                    # Trust it instead of re-decoding.
                    output_strs[i] = final_text[i]
                    continue

                toks = obj.output_tokens[i]
                if not toks:
                    continue
                try:
                    output_strs[i] = self.tokenizer.decode(
                        toks, skip_special_tokens=True
                    )
                except Exception as e:
                    elog(f"[detok] decode error rid={obj.rids[i]}: {e!r}")
                    errors[i] = f"decode error: {e!r}"

            vlog(f"[detok] send BatchStrOut rids={obj.rids} finished={obj.finished}")
            await self.send_to_tokenizer.send_pyobj(
                BatchStrOut(
                    rids=obj.rids,
                    output_strs=output_strs,
                    finished=obj.finished,
                    errors=errors,
                )
            )


def start_detokenizer_process(
    tokenizer_name: str,
    tokenizer_mode: str,
    trust_remote_code: bool,
    tokenizer_revision: Optional[str],
    detokenizer_port: int,
    tokenizer_port: int,
    pipe_writer,
) -> None:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        mgr = DetokenizerManager(
            tokenizer_name=tokenizer_name,
            detokenizer_port=detokenizer_port,
            tokenizer_port=tokenizer_port,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
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
    loop.run_until_complete(mgr.handle_loop())

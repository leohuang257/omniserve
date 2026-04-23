import asyncio
import dataclasses
import uuid
from typing import AsyncIterator, Dict, List, Optional

import zmq
import zmq.asyncio

from omniserve.sampling_params import SamplingParams
from omniserve.utils.tokenizer import get_tokenizer

from serving.io_struct import BatchStrOut, TokenizedGenerateReqInput
from serving.log_utils import elog, ilog, vlog


@dataclasses.dataclass
class ReqState:
    strip_prefix: str
    prev_text: str
    out_list: List[dict]
    finished: bool
    error: Optional[str]
    event: asyncio.Event


class TokenizerManager:
    def __init__(
        self,
        tokenizer_name: str,
        router_port: int,
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
        # tokenizer -> router
        self.send_to_router = ctx.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")
        # detokenizer -> tokenizer
        self.recv_from_detokenizer = ctx.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"tcp://127.0.0.1:{tokenizer_port}")

        self.rid_to_state: Dict[str, ReqState] = {}
        self._handle_loop_started = False
        self._handle_loop_lock = asyncio.Lock()

    async def _ensure_handle_loop(self) -> None:
        # create_task must run inside a running loop; defer to first request.
        if self._handle_loop_started:
            return
        async with self._handle_loop_lock:
            if self._handle_loop_started:
                return
            asyncio.get_running_loop().create_task(self._handle_loop())
            self._handle_loop_started = True

    async def generate_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        stream: bool = False,
        rid: Optional[str] = None,
        input_ids: Optional[List[int]] = None,
        strip_prefix: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        await self._ensure_handle_loop()

        rid = rid or uuid.uuid4().hex
        if input_ids is None:
            input_ids = self.tokenizer.encode(prompt)
        if strip_prefix is None:
            strip_prefix = prompt

        tokenized = TokenizedGenerateReqInput(
            rid=rid,
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=stream,
        )

        state = ReqState(
            strip_prefix=strip_prefix,
            prev_text="",
            out_list=[],
            finished=False,
            error=None,
            event=asyncio.Event(),
        )
        self.rid_to_state[rid] = state

        vlog(f"[tm] sending rid={rid} input_len={len(input_ids)}")
        await self.send_to_router.send_pyobj(tokenized)
        vlog(f"[tm] sent rid={rid}")

        try:
            while True:
                await state.event.wait()
                frames = state.out_list
                state.out_list = []
                state.event.clear()

                for frame in frames:
                    yield frame

                if state.finished:
                    break
        finally:
            self.rid_to_state.pop(rid, None)

    async def _handle_loop(self) -> None:
        ilog("[tm] handle_loop started")
        while True:
            try:
                obj: BatchStrOut = await self.recv_from_detokenizer.recv_pyobj()
            except Exception as e:
                elog(f"[tm] recv error: {e!r}")
                await asyncio.sleep(0.1)
                continue
            vlog(f"[tm] recv BatchStrOut rids={obj.rids} finished={obj.finished}")

            n = len(obj.rids)
            errors = obj.errors if obj.errors else [None] * n

            for i in range(n):
                rid = obj.rids[i]
                state = self.rid_to_state.get(rid)
                if state is None:
                    continue

                finished = bool(obj.finished[i])
                err = errors[i]
                full_text = obj.output_strs[i]

                if finished and full_text.startswith(state.strip_prefix):
                    full_text = full_text[len(state.strip_prefix):]

                if err is not None:
                    state.error = err
                    state.finished = True
                    state.out_list.append({
                        "text": state.prev_text,
                        "delta": "",
                        "finished": True,
                        "error": err,
                    })
                    state.event.set()
                    continue

                if full_text.startswith(state.prev_text):
                    delta = full_text[len(state.prev_text):]
                else:
                    delta = full_text

                if not finished and delta == "":
                    continue

                state.prev_text = full_text
                state.finished = finished
                state.out_list.append({
                    "text": full_text,
                    "delta": delta,
                    "finished": finished,
                    "error": None,
                })
                state.event.set()

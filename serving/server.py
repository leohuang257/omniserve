import argparse
import asyncio
import json
import multiprocessing as mp
import sys
import uuid
from typing import Optional

import uvicorn
import uvloop
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from omniserve.engine.arg_utils import EngineArgs
from omniserve.sampling_params import SamplingParams

from serving.detokenize_manager import start_detokenizer_process
from serving.log_utils import elog, ilog
from serving.openai_api import register_openai_routes
from serving.router import start_router_process
from serving.tokenize_manager import TokenizerManager


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


app = FastAPI()
tokenizer_manager: Optional[TokenizerManager] = None


def _build_sampling_params(req: GenerateRequest) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_new_tokens,
    )


@app.post("/generate")
async def generate(req: GenerateRequest):
    assert tokenizer_manager is not None, "tokenizer_manager not initialized"
    rid = uuid.uuid4().hex
    sampling_params = _build_sampling_params(req)

    if req.stream:
        async def sse_gen():
            try:
                async for frame in tokenizer_manager.generate_request(
                    prompt=req.prompt,
                    sampling_params=sampling_params,
                    stream=True,
                    rid=rid,
                ):
                    payload = {
                        "request_id": rid,
                        "delta": frame["delta"],
                        "text": frame["text"],
                        "finished": frame["finished"],
                        "error": frame.get("error"),
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                return

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    last = None
    async for frame in tokenizer_manager.generate_request(
        prompt=req.prompt,
        sampling_params=sampling_params,
        stream=False,
        rid=rid,
    ):
        last = frame

    if last is None:
        raise HTTPException(status_code=500, detail="no output produced")
    if last.get("error"):
        raise HTTPException(status_code=400, detail=last["error"])
    return {"request_id": rid, "text": last["text"], "finished": last["finished"]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/flush_cache")
async def flush_cache():
    # sglang's bench_serving calls this between warmup and the main run to
    # drop any prefix cache. We don't implement prefix caching, so it's a
    # no-op that exists only for protocol compatibility.
    return {"status": "ok", "note": "no prefix cache to flush"}


def _split_engine_args(argv):
    server_parser = argparse.ArgumentParser(add_help=False)
    server_parser.add_argument("--host", type=str, default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--router-port", type=int, default=28001)
    server_parser.add_argument("--tokenizer-port", type=int, default=28002)
    server_parser.add_argument("--detokenizer-port", type=int, default=28003)
    server_parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Name advertised under /v1/models and echoed back in OpenAI "
             "response bodies. Defaults to basename of --model.",
    )
    server_ns, remaining = server_parser.parse_known_args(argv)
    return remaining, server_ns


def _wait_child_ready(name: str, proc, parent_conn) -> None:
    ilog(f"[server] waiting for {name} to signal ready...")
    try:
        status, payload = parent_conn.recv()
    except EOFError:
        elog(f"[server] {name} died before signaling ready")
        proc.join()
        sys.exit(1)
    if status != "ok":
        elog(f"[server] {name} init failed:\n" + (payload or ""))
        proc.join()
        sys.exit(1)
    ilog(f"[server] {name} is ready")


def main():
    raw_argv = sys.argv[1:]
    engine_argv, server_ns = _split_engine_args(raw_argv)

    engine_parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(engine_parser)
    engine_ns = engine_parser.parse_args(engine_argv)

    tokenizer_name = engine_ns.tokenizer or engine_ns.model
    tokenizer_mode = getattr(engine_ns, "tokenizer_mode", "auto")
    trust_remote_code = getattr(engine_ns, "trust_remote_code", False)
    tokenizer_revision = getattr(engine_ns, "tokenizer_revision", None)

    ctx = mp.get_context("spawn")

    # 1) Start Detokenizer FIRST
    detok_parent, detok_child = ctx.Pipe(duplex=False)
    detok_proc = ctx.Process(
        target=start_detokenizer_process,
        args=(
            tokenizer_name,
            tokenizer_mode,
            trust_remote_code,
            tokenizer_revision,
            server_ns.detokenizer_port,
            server_ns.tokenizer_port,
            detok_child,
        ),
        daemon=True,
    )
    detok_proc.start()
    detok_child.close()
    _wait_child_ready("detokenizer", detok_proc, detok_parent)

    # 2) Start Router
    router_parent, router_child = ctx.Pipe(duplex=False)
    router_proc = ctx.Process(
        target=start_router_process,
        args=(
            engine_argv,
            server_ns.router_port,
            server_ns.detokenizer_port,
            router_child,
        ),
        daemon=True,
    )
    router_proc.start()
    router_child.close()
    _wait_child_ready("router", router_proc, router_parent)

    # 3) Build the TokenizerManager
    global tokenizer_manager
    tokenizer_manager = TokenizerManager(
        tokenizer_name=tokenizer_name,
        router_port=server_ns.router_port,
        tokenizer_port=server_ns.tokenizer_port,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
    )

    # 4) Register OpenAI-compatible routes
    import os
    served_model_name = server_ns.served_model_name or os.path.basename(
        (tokenizer_name or "model").rstrip("/")
    )
    register_openai_routes(app, tokenizer_manager, served_model_name=served_model_name)

    ilog(f"[server] listening on http://{server_ns.host}:{server_ns.port}")
    try:
        uvicorn.run(app, host=server_ns.host, port=server_ns.port, log_level="info")
    finally:
        for name, proc in [("router", router_proc), ("detokenizer", detok_proc)]:
            if proc.is_alive():
                ilog(f"[server] terminating {name}")
                proc.terminate()
                proc.join(timeout=5)


if __name__ == "__main__":
    main()

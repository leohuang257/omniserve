import argparse
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from omniserve.engine.arg_utils import EngineArgs
from omniserve.engine.llm_engine import LLMEngine
from omniserve.sampling_params import SamplingParams


# ---------- HTTP schema ----------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    text: str
    request_id: str


# ---------- globals ----------
engine: Optional[LLMEngine] = None
app = FastAPI()


# ---------- endpoint ----------
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    assert engine is not None, "engine not initialized"
    request_id = str(uuid.uuid4())

    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_new_tokens,
    )

    accepted = engine.add_request(
        request_id=request_id,
        prompt=req.prompt,
        sampling_params=sampling_params,
    )
    if not accepted:
        raise HTTPException(
            status_code=400,
            detail="prompt length exceeds max_model_len",
        )

    final_text = ""
    finished = False
    while engine.has_unfinished_requests():
        outputs = engine.step()
        if not outputs:
            break
        for out in outputs:
            if out["finished"]:
                final_text = out["text"]
                finished = True

    if not finished:
        raise HTTPException(status_code=500, detail="request did not finish")

    return GenerateResponse(text=final_text, request_id=request_id)


# ---------- startup ----------
def main():
    parser = argparse.ArgumentParser(
        description="Step 1 toy HTTP server wrapping LLMEngine."
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    global engine
    print("[server_v0] building engine (this may take a while)...")
    engine = LLMEngine.from_engine_args(engine_args)
    print(f"[server_v0] engine ready, listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
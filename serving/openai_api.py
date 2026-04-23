import json
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from omniserve.sampling_params import SamplingParams

from serving.log_utils import elog, ilog
from serving.tokenize_manager import TokenizerManager


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # vLLM / SGLang extension: don't let EOS stop generation. Useful for
    # benchmarking where you want exactly `max_tokens` output tokens.
    ignore_eos: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    echo: Optional[bool] = False
    user: Optional[str] = None
    ignore_eos: Optional[bool] = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


_CHAT_STOP_TOKEN_STRINGS = [
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
]


def _resolve_chat_stop_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    seen = set()
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
        seen.add(tokenizer.eos_token_id)
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    for s in _CHAT_STOP_TOKEN_STRINGS:
        tid = vocab.get(s)
        if tid is not None and tid not in seen:
            ids.append(tid)
            seen.add(tid)
    return ids


def _strip_trailing_specials(text: str) -> str:
    """Drop trailing chat-stop special-token strings from a decoded text.

    The engine's output_text is decode(all_ids) without skip_special_tokens,
    so if the model emitted e.g. <|eot_id|> before stopping, it will appear
    as literal text at the tail. Remove it so clients see clean assistant text.
    """
    changed = True
    while changed:
        changed = False
        for s in _CHAT_STOP_TOKEN_STRINGS:
            if text.endswith(s):
                text = text[: -len(s)]
                changed = True
        text = text.rstrip()
    return text


def _build_sampling_params(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    ignore_eos: bool = False,
) -> SamplingParams:
    if isinstance(stop, str):
        stop_list = [stop]
    elif stop is None:
        stop_list = []
    else:
        stop_list = list(stop)
    return SamplingParams(
        temperature=max(temperature, 0.0),
        top_p=top_p if 0.0 < top_p <= 1.0 else 1.0,
        top_k=top_k if top_k == -1 or top_k >= 1 else -1,
        max_tokens=max_tokens,
        stop=stop_list,
        stop_token_ids=stop_token_ids or [],
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        ignore_eos=ignore_eos,
    )


def _include_usage(stream_options: Optional[Dict[str, Any]]) -> bool:
    return bool(stream_options and stream_options.get("include_usage", False))


def _now() -> int:
    return int(time.time())


# ============================================================================
# route registration
# ============================================================================
def register_openai_routes(
    app: FastAPI,
    tokenizer_manager: TokenizerManager,
    served_model_name: str,
) -> None:
    """Attach /v1/models, /v1/completions, /v1/chat/completions to `app`.

    Call once, after tokenizer_manager has been constructed in the parent.
    """
    tokenizer = tokenizer_manager.tokenizer
    chat_stop_token_ids = _resolve_chat_stop_token_ids(tokenizer)
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    ilog(
        f"[openai_api] served_model_name={served_model_name!r} "
        f"chat_template={'yes' if has_chat_template else 'NO'} "
        f"chat_stop_ids={chat_stop_token_ids}"
    )

    # ------------------------------------------------------------------ models
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": _now(),
                    "owned_by": "omniserve",
                }
            ],
        }

    # ------------------------------------------------------------------ completions
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if isinstance(req.prompt, list):
            # OpenAI allows batching multiple prompts here. We support only one
            # for simplicity; batch via concurrent HTTP calls instead.
            if len(req.prompt) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="batched prompts in one request are not supported; "
                           "issue N concurrent /v1/completions calls instead.",
                )
            prompt_text = req.prompt[0]
        else:
            prompt_text = req.prompt

        rid = f"cmpl-{uuid.uuid4().hex}"
        sp = _build_sampling_params(
            max_tokens=req.max_tokens or 128,
            temperature=req.temperature if req.temperature is not None else 1.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            top_k=req.top_k if req.top_k is not None else -1,
            stop=req.stop,
            presence_penalty=req.presence_penalty or 0.0,
            frequency_penalty=req.frequency_penalty or 0.0,
            ignore_eos=bool(req.ignore_eos),
        )
        prompt_tokens = len(tokenizer.encode(prompt_text))
        include_usage = _include_usage(req.stream_options)

        if req.stream:
            async def sse():
                accum = ""
                try:
                    async for frame in tokenizer_manager.generate_request(
                        prompt=prompt_text,
                        sampling_params=sp,
                        stream=True,
                        rid=rid,
                    ):
                        if frame.get("error"):
                            yield f"data: {json.dumps(_completion_chunk(rid, served_model_name, text='', finish_reason='error'))}\n\n"
                            break
                        delta_text = frame["delta"] or ""
                        accum += delta_text
                        if frame["finished"]:
                            yield f"data: {json.dumps(_completion_chunk(rid, served_model_name, text=delta_text, finish_reason='stop'))}\n\n"
                        else:
                            yield f"data: {json.dumps(_completion_chunk(rid, served_model_name, text=delta_text, finish_reason=None))}\n\n"

                    if include_usage:
                        completion_tokens = len(
                            tokenizer.encode(accum, add_special_tokens=False)
                        ) if accum else 0
                        usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ).dict()
                        yield f"data: {json.dumps(_usage_only_completion_chunk(rid, served_model_name, usage))}\n\n"
                except Exception as e:
                    elog(f"[openai_api] /v1/completions stream error: {e!r}")
                finally:
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # non-streaming
        last = None
        async for frame in tokenizer_manager.generate_request(
            prompt=prompt_text,
            sampling_params=sp,
            stream=False,
            rid=rid,
        ):
            last = frame
        if last is None:
            raise HTTPException(status_code=500, detail="no output produced")
        if last.get("error"):
            raise HTTPException(status_code=400, detail=last["error"])

        text_out = last["text"]
        if req.echo:
            text_out = prompt_text + text_out
        completion_tokens = len(tokenizer.encode(text_out)) if text_out else 0
        return {
            "id": rid,
            "object": "text_completion",
            "created": _now(),
            "model": served_model_name,
            "choices": [
                {
                    "text": text_out,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ).dict(),
        }

    # ------------------------------------------------------------------ chat
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not has_chat_template:
            raise HTTPException(
                status_code=400,
                detail="this tokenizer has no chat_template; use /v1/completions instead",
            )

        # Render + tokenize with the tokenizer's own Jinja2 template.
        messages = [m.dict(exclude_none=True) for m in req.messages]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"chat template render failed: {e!r}"
            )

        # strip_prefix must match how the engine will decode. The engine does
        # tokenizer.decode(all_ids) without skip_special_tokens, so use the
        # same call on input_ids.
        strip_prefix = tokenizer.decode(input_ids)

        rid = f"chatcmpl-{uuid.uuid4().hex}"
        sp = _build_sampling_params(
            max_tokens=req.max_tokens or 256,
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 1.0,
            top_k=req.top_k if req.top_k is not None else -1,
            stop=req.stop,
            stop_token_ids=chat_stop_token_ids,
            presence_penalty=req.presence_penalty or 0.0,
            frequency_penalty=req.frequency_penalty or 0.0,
            ignore_eos=bool(req.ignore_eos),
        )
        prompt_tokens = len(input_ids)
        include_usage = _include_usage(req.stream_options)

        if req.stream:
            async def sse():
                first = _chat_chunk(
                    rid, served_model_name,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None,
                )
                yield f"data: {json.dumps(first)}\n\n"

                accum = ""
                try:
                    async for frame in tokenizer_manager.generate_request(
                        prompt=prompt_text,
                        sampling_params=sp,
                        stream=True,
                        rid=rid,
                        input_ids=input_ids,
                        strip_prefix=strip_prefix,
                    ):
                        if frame.get("error"):
                            yield f"data: {json.dumps(_chat_chunk(rid, served_model_name, delta={}, finish_reason='error'))}\n\n"
                            break

                        if frame["finished"]:
                            clean_delta = _strip_trailing_specials(frame["delta"])
                            if clean_delta:
                                accum += clean_delta
                                yield f"data: {json.dumps(_chat_chunk(rid, served_model_name, delta={'content': clean_delta}, finish_reason=None))}\n\n"
                            yield f"data: {json.dumps(_chat_chunk(rid, served_model_name, delta={}, finish_reason='stop'))}\n\n"
                        else:
                            piece = frame["delta"] or ""
                            if not piece:
                                continue
                            accum += piece
                            yield f"data: {json.dumps(_chat_chunk(rid, served_model_name, delta={'content': piece}, finish_reason=None))}\n\n"

                    if include_usage:
                        completion_tokens = len(
                            tokenizer.encode(accum, add_special_tokens=False)
                        ) if accum else 0
                        usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ).dict()
                        yield f"data: {json.dumps(_usage_only_chat_chunk(rid, served_model_name, usage))}\n\n"
                except Exception as e:
                    elog(f"[openai_api] /v1/chat/completions stream error: {e!r}")
                finally:
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # non-streaming
        last = None
        async for frame in tokenizer_manager.generate_request(
            prompt=prompt_text,
            sampling_params=sp,
            stream=False,
            rid=rid,
            input_ids=input_ids,
            strip_prefix=strip_prefix,
        ):
            last = frame
        if last is None:
            raise HTTPException(status_code=500, detail="no output produced")
        if last.get("error"):
            raise HTTPException(status_code=400, detail=last["error"])

        content = _strip_trailing_specials(last["text"])
        completion_tokens = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0
        return {
            "id": rid,
            "object": "chat.completion",
            "created": _now(),
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ).dict(),
        }


# ============================================================================
# chunk builders
# ============================================================================
def _completion_chunk(
    rid: str,
    model: str,
    text: str,
    finish_reason: Optional[str],
    usage: Optional[dict] = None,
) -> dict:
    return {
        "id": rid,
        "object": "text_completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _chat_chunk(
    rid: str,
    model: str,
    delta: dict,
    finish_reason: Optional[str],
    usage: Optional[dict] = None,
) -> dict:
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _usage_only_chat_chunk(rid: str, model: str, usage: dict) -> dict:
    """Terminal chunk when stream_options.include_usage=True (OpenAI spec)."""
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [],
        "usage": usage,
    }


def _usage_only_completion_chunk(rid: str, model: str, usage: dict) -> dict:
    return {
        "id": rid,
        "object": "text_completion",
        "created": _now(),
        "model": model,
        "choices": [],
        "usage": usage,
    }

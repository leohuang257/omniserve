"""Gradio chat UI over the OpenAI-compatible /v1/chat/completions endpoint.

Streaming, multi-turn, with temperature / max_tokens / top_p / system-prompt controls.

Run it AFTER the server is up:

    # Terminal 1
    python -m serving.server --model $MODEL_PATH ...
    # Terminal 2
    python gradio_ui.py

Env vars:
    SERVER_URL   base URL (default http://127.0.0.1:8000)
    MODEL_NAME   model name to send (default llama-3-8b-qserve)
    GRADIO_HOST  UI bind host (default 127.0.0.1)
    GRADIO_PORT  UI bind port (default 7860)

See `gradio_completion.py` for the older raw /generate playground.
"""
import json
import os
import time
from typing import Iterator, List, Tuple

import gradio as gr
import httpx


DEFAULT_BASE = os.environ.get("SERVER_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "llama-3-8b-qserve")


def _build_messages(
    history: List[Tuple[str, str]],
    user_msg: str,
    system_prompt: str,
) -> List[dict]:
    msgs: List[dict] = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    for u, a in history:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def _chat_stream(
    base_url: str,
    model: str,
    history: List[Tuple[str, str]],
    user_msg: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Iterator[Tuple[List[Tuple[str, str]], str]]:
    """Yields (updated_history, status_line) pairs."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": _build_messages(history, user_msg, system_prompt),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": True,
    }

    new_history = history + [(user_msg, "")]
    yield new_history, "… connecting"

    t0 = time.perf_counter()
    t_first = 0.0
    n_tokens = 0
    reply = ""

    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    err_body = resp.read().decode("utf-8", "replace")
                    new_history[-1] = (user_msg, f"**HTTP {resp.status_code}**: {err_body[:400]}")
                    yield new_history, f"HTTP {resp.status_code}"
                    return

                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = (choice.get("delta") or {}).get("content", "") or ""
                    finish = choice.get("finish_reason")
                    if delta:
                        if not t_first:
                            t_first = time.perf_counter()
                        reply += delta
                        n_tokens += 1
                        new_history[-1] = (user_msg, reply)
                        yield new_history, _status(t0, t_first, n_tokens)
                    if finish is not None:
                        break
    except Exception as e:
        new_history[-1] = (user_msg, f"**client error**: {e!r}")
        yield new_history, f"client error: {e!r}"
        return

    yield new_history, _status(t0, t_first, n_tokens, final=True)


def _status(t0: float, t_first: float, n_tokens: int, final: bool = False) -> str:
    now = time.perf_counter()
    parts = [f"elapsed {(now - t0) * 1000:.0f} ms"]
    if t_first:
        parts.append(f"TTFT {(t_first - t0) * 1000:.0f} ms")
        if now > t_first:
            parts.append(f"{n_tokens / (now - t_first):.1f} chunk/s")
    parts.append(f"{n_tokens} chunks")
    if final:
        parts.append("✓ done")
    return "  ·  ".join(parts)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OmniServe Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# OmniServe Chat\n"
            "Streaming chat over the OpenAI-compatible "
            "`/v1/chat/completions` endpoint."
        )

        with gr.Row():
            base_url = gr.Textbox(label="Server base URL", value=DEFAULT_BASE, scale=3)
            model_name = gr.Textbox(label="Model", value=DEFAULT_MODEL, scale=2)

        chatbot = gr.Chatbot(height=420, label="Conversation")
        msg = gr.Textbox(
            label="Your message",
            placeholder="Say hi...",
            lines=2,
        )
        status = gr.Markdown("")

        with gr.Accordion("Generation settings", open=False):
            system_prompt = gr.Textbox(
                label="System prompt",
                value="You are a helpful assistant.",
                lines=2,
            )
            with gr.Row():
                max_tokens = gr.Slider(8, 2048, value=256, step=8, label="max_tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.01, label="top_p")

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        def _submit(user_msg, history, base_url, model, sys_p, mx, t, tp):
            if not user_msg.strip():
                yield history, "", ""
                return
            for hist, st in _chat_stream(base_url, model, history, user_msg, sys_p, mx, t, tp):
                yield hist, "", st

        send_btn.click(
            _submit,
            inputs=[msg, chatbot, base_url, model_name, system_prompt,
                    max_tokens, temperature, top_p],
            outputs=[chatbot, msg, status],
        )
        msg.submit(
            _submit,
            inputs=[msg, chatbot, base_url, model_name, system_prompt,
                    max_tokens, temperature, top_p],
            outputs=[chatbot, msg, status],
        )
        clear_btn.click(lambda: ([], "", ""), None, [chatbot, msg, status])

    return demo


if __name__ == "__main__":
    host = os.environ.get("GRADIO_HOST", "127.0.0.1")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo = build_ui()
    demo.queue()
    demo.launch(server_name=host, server_port=port, show_api=False)

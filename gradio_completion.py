"""Gradio front-end for the serving/server /generate endpoint.

A thin UI over the HTTP API. Handles streaming (SSE) and non-streaming.
Run it AFTER the server is up:

    # Terminal 1: start the server
    python -m serving.server --model $MODEL_PATH ...

    # Terminal 2: start the UI (default reads SERVER_URL or falls back)
    python gradio_ui.py

Then open http://127.0.0.1:7860 in a browser.

Env vars:
    SERVER_URL   full URL to /generate (default http://127.0.0.1:8000/generate)
    GRADIO_HOST  host to bind the UI (default 127.0.0.1)
    GRADIO_PORT  port to bind the UI (default 7860)
"""
import json
import os
import time
from typing import Iterator, Tuple

import gradio as gr
import httpx


DEFAULT_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8000/generate")


def _fmt_stats(
    start: float,
    first: float,
    now: float,
    n_chunks: int,
    out_tokens_est: int,
) -> str:
    elapsed = now - start
    ttft = (first - start) if first else None
    tps = (out_tokens_est / (now - first)) if first and (now - first) > 0 else None
    parts = [f"elapsed {elapsed*1000:.0f} ms"]
    if ttft is not None:
        parts.append(f"TTFT {ttft*1000:.0f} ms")
    if tps is not None:
        parts.append(f"{tps:.1f} tok/s (approx)")
    parts.append(f"{n_chunks} chunks")
    return "  |  ".join(parts)


def _generate_stream(
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Iterator[Tuple[str, str]]:
    """Yields (text_so_far, stats_line) tuples for the streaming UI."""
    if not prompt.strip():
        yield ("", "prompt is empty")
        return

    payload = {
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": True,
    }

    t0 = time.perf_counter()
    t_first = 0.0
    n_chunks = 0
    text = ""

    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", "replace")
                    yield ("", f"HTTP {resp.status_code}: {body[:400]}")
                    return
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    now = time.perf_counter()
                    if not t_first:
                        t_first = now
                    try:
                        data = json.loads(line[len("data:"):].strip())
                    except json.JSONDecodeError:
                        continue
                    n_chunks += 1
                    if data.get("error"):
                        yield (text, f"error: {data['error']}")
                        return
                    text = data.get("text", text)
                    # rough token estimate for on-the-fly TPS readout
                    out_tokens_est = max(1, len(text[len(prompt):]) // 4)
                    stats = _fmt_stats(t0, t_first, now, n_chunks, out_tokens_est)
                    yield (text, stats)
                    if data.get("finished"):
                        return
    except Exception as e:
        yield (text, f"client error: {e!r}")


def _generate_nonstream(
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    if not prompt.strip():
        return ("", "prompt is empty")
    payload = {
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(url, json=payload)
            dt = time.perf_counter() - t0
            if resp.status_code != 200:
                return ("", f"HTTP {resp.status_code}: {resp.text[:400]}")
            body = resp.json()
            text = body.get("text", "")
            out_tokens_est = max(1, len(text[len(prompt):]) // 4)
            tps = out_tokens_est / dt if dt > 0 else 0
            return (text, f"elapsed {dt*1000:.0f} ms  |  ~{tps:.1f} tok/s")
    except Exception as e:
        return ("", f"client error: {e!r}")


def _handle(
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
) -> Iterator[Tuple[str, str]]:
    if stream:
        yield from _generate_stream(
            url, prompt, max_new_tokens, temperature, top_p
        )
    else:
        yield _generate_nonstream(
            url, prompt, max_new_tokens, temperature, top_p
        )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OmniServe Playground") as demo:
        gr.Markdown("# OmniServe Playground")
        gr.Markdown(
            "Thin HTTP client over the local `/generate` endpoint "
            "(FastAPI + TokenizerManager + Router + DetokenizerManager)."
        )

        with gr.Row():
            server_url = gr.Textbox(
                label="Server URL",
                value=DEFAULT_URL,
                scale=4,
            )
            stream_box = gr.Checkbox(label="Streaming", value=True, scale=1)

        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            placeholder="The capital of France is",
            value="The capital of France is",
        )

        with gr.Row():
            max_new = gr.Slider(1, 1024, value=128, step=1, label="Max new tokens")
            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.05, 1.0, value=1.0, step=0.01, label="Top-p")

        with gr.Row():
            submit = gr.Button("Generate", variant="primary")
            clear = gr.Button("Clear")

        output = gr.Textbox(
            label="Output",
            lines=14,
            show_copy_button=True,
            interactive=False,
        )
        stats = gr.Markdown("")

        submit.click(
            fn=_handle,
            inputs=[server_url, prompt, max_new, temperature, top_p, stream_box],
            outputs=[output, stats],
        )
        clear.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[output, stats],
        )

    return demo


if __name__ == "__main__":
    host = os.environ.get("GRADIO_HOST", "127.0.0.1")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo = build_ui()
    demo.queue()  # so generators can stream to multiple clients
    demo.launch(server_name=host, server_port=port, show_api=False)

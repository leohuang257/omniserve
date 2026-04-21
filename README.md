# Online Serving for OmniServe (W4A8KV4 Quantized LLMs)

Building an online, concurrent HTTP serving layer on top of
[OmniServe / QServe](https://github.com/mit-han-lab/qserve)'s existing
quantized inference engine. Architecture inspired by SGLang's multi-process
design with ZMQ-based inter-process communication.

## Motivation

OmniServe ships a high-performance offline batched inference engine for
W4A8KV4-quantized LLMs, but has no online serving frontend. This project
adds one: an HTTP API, streaming support, OpenAI-compatible endpoints,
and a Gradio demo — all reusing OmniServe's existing `LLMEngine`, scheduler,
and kernels unchanged.

The final deliverable is a Gradio demo that serves a QServe-quantized
Llama-3-8B-Instruct model with real-time streaming.

## Target Architecture

```
┌──────────────────┐      ZMQ      ┌────────────────────┐      ZMQ      ┌────────────────────┐
│  tokenizer_mgr   │ ───PUSH────>  │  router_manager    │ ───PUSH────>  │  detokenizer_mgr   │
│  (main process:  │               │  (engine process:  │               │  (decode process)  │
│   FastAPI +      │ <──PULL────── │   LLMEngine.step   │               │                    │
│   tokenize)      │               │   single-threaded) │               │                    │
└──────────────────┘               └────────────────────┘               └────────────────────┘
```

Three processes, two ZMQ channels. The HTTP handler never touches the engine
directly — it submits tokenized requests over ZMQ and awaits results keyed by
`request_id`. The engine process owns a single-threaded `step()` loop, which
is what allows true in-flight batching across concurrent HTTP requests.

## Current Status

**Step 1: single-process toy server — ✅ complete**  
**Step 2: splitting into two processes with ZMQ — 🔨 in progress**

| Step | Goal | Files | Status |
|------|------|-------|--------|
| 0 | Learn ZMQ with toy PUSH/PULL scripts | `toy_sender.py`, `toy_receiver.py`, `toy_protocol.py` | ✅ done |
| 1 | Wrap `LLMEngine` behind a synchronous FastAPI endpoint | `omniserve/serving/server_v0.py` | ✅ done |
| 2 | Split server + engine into two processes, ZMQ between them | `io_struct.py`, `server.py`, `router_manager.py` | 🔨 next |
| 3 | Add a dedicated detokenizer process | `detokenizer_manager.py` | ⏳ |
| 4 | Token-level streaming (SSE) | extends `server.py` | ⏳ |
| 5 | OpenAI-compatible `/v1/completions` + Gradio demo | `openai_protocol.py`, `gradio_app.py` | ⏳ |
| 6 | Benchmark (throughput, latency, concurrency) | `benchmarks/bench_serving.py` | ⏳ |

### Step 1 — what works

- `omniserve/serving/server_v0.py` boots an `LLMEngine` in-process and exposes `POST /generate`.
- Sequential requests return coherent output with the Llama-3-8B-Instruct W4A8KV4 checkpoint (~0.3s / 32 tokens).
- All of OmniServe's `EngineArgs` CLI flags are reused verbatim via `EngineArgs.add_cli_args(parser)` — same launch flags as offline scripts like `qserve_e2e_generation.py`.

### Step 1 — known limitations (which motivate Step 2)

Running three `curl` requests concurrently against `server_v0.py` produces:

- two `500 Internal Server Error` responses
- one response that mixes tokens from multiple requests (`"Request 3::: The request33 is33 a3 11"`)
- total wall time *longer* than the sequential case

Root cause: the FastAPI handler is synchronous and each invocation drives its
own `while engine.has_unfinished_requests(): engine.step()` loop. When multiple
requests arrive at once, several threads concurrently mutate engine state and
race to consume each other's outputs. The engine is not thread-safe, and even
if it were, the handler has no way to tell which `output["id"]` corresponds
to its own request.

Step 2 fixes this by moving `engine.step()` into a single-threaded dedicated
process and reducing the HTTP handler's job to "submit over ZMQ, await my
result by `request_id`".

## Running the toy server (Step 1)

Requires a QServe-quantized checkpoint at `$MODEL_PATH`. Same env vars and
flags as OmniServe's offline inference scripts:

```bash
MODEL_PATH=./qserve_checkpoints/Llama-3-8B-Instruct-QServe

NUM_RETRIEVAL_GPU_PAGE_BLOCKS=12000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=0 \
CHUNK_PREFILL_SIZE=2147000000 \
python -m omniserve.serving.server_v0 \
  --model $MODEL_PATH \
  --quant-path $MODEL_PATH \
  --precision w4a8kv4 \
  --group-size -1 \
  --ifb-mode \
  --kv-quant-granularity fine_grained \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4195000 \
  --chunk-prefill-size 1024000 \
  --sparse-decode-mode 0
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"The capital of France is","max_new_tokens":32}'
```

## Project Layout

```
omniserve/
├── engine/              # existing: LLMEngine, EngineArgs (unchanged)
├── core/                # existing: scheduler, block manager (unchanged)
├── worker/              # existing: model runner, kernels (unchanged)
├── modeling/            # existing: quantized model definitions (unchanged)
└── serving/             # NEW: online serving layer
    ├── __init__.py
    └── server_v0.py     # Step 1: single-process toy server
```

Planned additions under `omniserve/serving/`:

- `io_struct.py` — shared dataclasses for cross-process messages
- `server.py` — production FastAPI server (async, non-blocking)
- `router_manager.py` — engine subprocess: ZMQ loop around `LLMEngine.step()`
- `detokenizer_manager.py` — decode subprocess
- `openai_protocol.py` — OpenAI-compatible request/response schemas
- `gradio_app.py` — demo UI

## Design Principles

1. **Don't touch the engine.** The model, kernels, scheduler, KV cache, and
   quantization paths are all already better in OmniServe than in the
   reference (nano-sglang). This project adds a serving layer on top and
   changes nothing below it.

2. **Every step produces something runnable.** No big-bang integration.
   Each step ends with a command you can `curl` or `python` to verify.

3. **Scope cuts kept explicit** (not in scope for this project):
   - Tensor parallelism / multi-GPU (single-GPU only for now)
   - RadixCache-style prefix reuse (OmniServe's existing `PrefixPool` is used as-is)
   - Multimodal (LLaVA) and MoE (Mixtral)
   - Constrained/FSM decoding

## References

- [QServe paper](https://arxiv.org/abs/2405.04532) (Lin et al., 2024)
- [OmniServe / QServe code](https://github.com/mit-han-lab/qserve)
- [SGLang](https://github.com/sgl-project/sglang) — architecture reference for the multi-process serving design
- [vLLM](https://github.com/vllm-project/vllm) — original source of much of OmniServe's engine API
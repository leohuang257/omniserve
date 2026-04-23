# OmniServe-SGL

【English | [中文](./README_zh.md)】

A small, readable, SGLang-style serving stack built around the
[OmniServe / QServe](https://github.com/mit-han-lab/omniserve) quantized LLM
engine. Three cooperating processes, ZMQ IPC, streaming
[OpenAI-compatible](https://platform.openai.com/docs/api-reference) HTTP API,
ready to benchmark with SGLang's official
[`bench_serving.py`](https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py),
plus a chat UI.

---

## Why this repo

This started as **Project 12 ("Online Quantized LLM Serving with QServe")** from
MIT's [6.5940 TinyML and Efficient Deep Learning Computing (Fall 2024)](https://efficientml.ai/)
course. The project brief, by the QServe authors (Haotian Tang, Shang Yang,
Yujun Lin), points out that the open-source
[QServe](https://github.com/mit-han-lab/qserve) release only ships an
*offline* batched generation script (`qserve_e2e_generation.py`), while
real-world deployments need an *online* serving system — something like
[vLLM](https://github.com/vllm-project/vllm) or
[SGLang](https://github.com/sgl-project/sglang), with a Gradio demo on top.
I picked it up as a way to learn MLSys / AI infra hands-on.

Concretely, `omniserve/qserve_e2e_generation.py` shows the engine can generate
text, but it runs as a single blocking Python script. A real serving stack
needs:

- concurrent HTTP requests without blocking the engine step loop,
- streaming output (SSE) per request,
- OpenAI-compatible endpoints so existing SDKs / evaluators work unchanged,
- clean separation of CPU-bound work (tokenize / detokenize) from GPU-bound
  work (engine step), each with its own event loop.

This project takes the minimal amount of code to get there, loosely following
SGLang's three-process layout so the architecture is easy to explain.

QServe's W4A8KV4 quantization (the "QoQ" algorithm from
[arXiv:2405.04532](https://arxiv.org/abs/2405.04532)) is the quantized
backend we're serving; all of the quantization kernels, attention backends
and model code in `omniserve/` and `kernels/` are upstream and untouched.

---

## How it works: one request, end to end

The serving stack has three cooperating processes connected by ZMQ PUSH
sockets. Each runs its own `uvloop` event loop.

```
           HTTP                      ZMQ PUSH                  ZMQ PUSH
  client ────────►  FastAPI + TokenizerManager ─────►  Router ─────────►  DetokenizerManager
                          (P0: CPU)                  (P1: GPU/engine)          (P2: CPU)
                              ▲                                                       │
                              └────────────── ZMQ PUSH ( BatchStrOut ) ────────────────┘
```

### High-level lifecycle

Following one `/v1/chat/completions` request from input to output:

1. User starts the server — the parent process spawns the
   **DetokenizerManager** (P2) and the **Router** (P1) as children, waits
   for each child's "ok" signal over `multiprocessing.Pipe`, constructs the
   **TokenizerManager** (P0) in-process, registers the OpenAI-compatible
   routes on the FastAPI app, and starts Uvicorn. Each process then runs an
   infinite `asyncio`/`uvloop` event loop.

2. User sends a `POST /v1/chat/completions` request. Uvicorn dispatches it
   to the `chat_completions` route registered in `openai_api.py`.

3. `chat_completions` parses the body into a `ChatCompletionRequest`,
   renders the chat history through the HF tokenizer's
   `apply_chat_template(..., add_generation_prompt=True)` to get both a
   prompt string and the corresponding `input_ids`, computes
   `strip_prefix = tokenizer.decode(input_ids)`, builds a `SamplingParams`
   (with the model's chat stop-token ids auto-injected), and calls
   `TokenizerManager.generate_request(...)`.

4. `TokenizerManager.generate_request` wraps the request in a
   `TokenizedGenerateReqInput(rid, input_ids, sampling_params, stream)`,
   sends it to the Router over its PUSH socket, allocates a local
   `ReqState(strip_prefix, prev_text="", out_list=[], event=asyncio.Event())`
   in `rid_to_state[rid]`, then loops `await state.event.wait()` to yield
   each new frame back to the HTTP handler.

5. The Router's event loop (running alongside its `LLMEngine` in P1):
   - `loop_for_recv_requests` receives `TokenizedGenerateReqInput`s from
     ZMQ and appends them to `recv_reqs`.
   - `loop_for_forward` each tick:
     - `_admit_new_requests()` — for each pending request, call
       `llm_engine.add_request(input_ids, sampling_params, ...)` which
       returns an internal integer `seq_id`, and record
       `seq_id_to_rid[seq_id] = rid` plus `seq_id_to_input_len[seq_id]`.
     - `outs = self.llm_engine.step()` — one forward step: prefill for
       newly admitted sequences, one decode token for running sequences,
       all fused into a single GPU call (continuous batching / IFB).
     - `_send_step_outputs(outs)` — translate each `seq_id → rid`, strip
       the prompt prefix from cumulative tokens, and ship a
       `BatchTokenIDOut` to the DetokenizerManager over ZMQ.

6. The DetokenizerManager's `handle_loop` receives `BatchTokenIDOut`,
   either (a) reuses the authoritative `final_text` from the engine when a
   request has finished, or (b) re-decodes the cumulative generated tokens
   for intermediate frames; wraps the result in `BatchStrOut(rids,
   output_strs, finished)`, and pushes it back to the TokenizerManager.

7. The TokenizerManager's background `handle_loop` receives `BatchStrOut`,
   strips `state.strip_prefix` from the full text (so the chat preamble
   never leaks into output), computes `delta = full_text[len(prev_text):]`,
   appends `{text, delta, finished, error}` to `state.out_list`, updates
   `prev_text`, and calls `state.event.set()` — which wakes the HTTP
   handler waiting inside `generate_request`.

8. `chat_completions` converts each frame into an OpenAI-shaped SSE chunk
   (`data: {...}\n\n`) and yields it through `StreamingResponse`. On the
   finished frame it emits a `finish_reason="stop"` chunk, optionally a
   `usage`-only terminator chunk when the client sent
   `stream_options.include_usage=true`, then `data: [DONE]\n\n`. Uvicorn
   streams each chunk to the client over the open HTTP connection.

### Starting the server (`serving/server.py`)

`server.py` is the process orchestrator. `main()` runs in the parent and:

1. Parses CLI via `argparse.parse_known_args`. Flags that belong to
   `LLMEngine` (e.g. `--model`, `--precision`, `--max-num-seqs`, …) are
   bundled into an `EngineArgs` and forwarded to the Router's child
   process; serving-layer flags (`--host`, `--port`, `--router-port`,
   `--detokenizer-port`, `--served-model-name`) stay in the parent.
2. Calls `mp.get_context("spawn").Process(target=start_detokenizer_process,
   args=(detokenizer_port, tokenizer_port, model_path, pipe_writer))` to
   fork P2, then calls `_wait_child_ready(pipe_reader)` which blocks until
   the child writes `"ok"` (or the exception traceback) back up the pipe.
3. Repeats for P1: `Process(target=start_router_process, args=(engine_args,
   router_port, detokenizer_port, pipe_writer))`. P1 is slow to come
   ready — `LLMEngine.from_engine_args(...)` does weight loading, KV
   cache allocation and CUDA kernel warmup before it signals `"ok"`
   (usually ~30–90 s for an 8 B quantized model).
4. Constructs `TokenizerManager` in the parent process. It binds its own
   ZMQ sockets (PUSH to Router, PULL from DetokenizerManager) and
   schedules a background `handle_loop` task on Uvicorn's event loop.
5. Calls `register_openai_routes(app, tokenizer_manager, served_model_name)`
   to attach `/v1/models`, `/v1/completions`, `/v1/chat/completions`.
   The native `/generate` and `/flush_cache` routes are registered
   directly on `app` in `server.py`.
6. Runs `uvicorn.run(app, host=..., port=...)`. A `finally` block
   terminates both children when the parent exits.

`spawn` (not `fork`) is used so the Router's child inherits no open CUDA
context from the parent, avoiding double-init of the GPU driver.

### Dispatching HTTP requests (`serving/openai_api.py`)

`register_openai_routes` attaches three endpoints:

- `GET /v1/models` — returns the `served_model_name` from `--served-model-name`.
- `POST /v1/completions` — raw text completion. Supports `prompt`,
  `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, `stream_options`,
  `echo`, `ignore_eos`.
- `POST /v1/chat/completions` — chat completion with streaming support.

For chat completions specifically:

1. The incoming JSON is parsed into a `ChatCompletionRequest` (Pydantic).
2. `tokenizer.apply_chat_template(messages, tokenize=False,
   add_generation_prompt=True)` renders the chat history into a single
   prompt string; a second call with `tokenize=True` returns matching
   `input_ids`. Both are computed so we can trust the engine to see
   exactly the same ids we measure on the HTTP side.
3. `strip_prefix = tokenizer.decode(input_ids)` is the string the engine's
   own `tokenizer.decode(all_ids)` would produce for the prompt portion
   — stripping this prefix from the engine's final text (in step 7 above)
   gives clean assistant output.
4. `_build_sampling_params(...)` creates a `SamplingParams` and injects
   `chat_stop_token_ids` (resolved once at startup by
   `_resolve_chat_stop_token_ids(tokenizer)` — looks up `<|eot_id|>`,
   `<|im_end|>`, `</s>`, etc. in the vocab). If the client set
   `ignore_eos=true`, it is forwarded to the engine.
5. Streaming: the route `async for`s over
   `tokenizer_manager.generate_request(..., input_ids=input_ids,
   strip_prefix=strip_prefix)` and converts each frame into a chat chunk
   via `_chat_chunk(...)`. The terminal frame gets a `finish_reason="stop"`
   chunk; when `stream_options.include_usage` is true, an extra
   `{choices: [], usage: {...}}` terminator chunk is yielded before
   `[DONE]` (OpenAI spec).
6. Non-streaming: the route awaits the last frame, runs
   `_strip_trailing_specials` on the text (to drop any literal special-
   token strings that survived decoding), and returns a
   `ChatCompletion`-shaped JSON with computed `usage`.

### TokenizerManager (`serving/tokenize_manager.py`)

Runs in-process with FastAPI. It owns the per-request state.

**Initialization**
- Creates two `zmq.asyncio` sockets: PUSH to Router, PULL from
  DetokenizerManager.
- Loads the HF `AutoTokenizer`.
- Schedules `handle_loop` as a background task on the event loop.
- Maintains `rid_to_state: Dict[str, ReqState]`.

**`generate_request(prompt, sampling_params, stream, rid, input_ids=None,
strip_prefix=None)`** — called by every HTTP handler.

1. If `input_ids` was not passed in, tokenize `prompt` with HF
   `tokenizer.encode`. (Chat uses `apply_chat_template` in the route, so
   the route passes both `input_ids` and `strip_prefix` in directly.)
2. Build `TokenizedGenerateReqInput(rid, input_ids, sampling_params,
   stream)` and send it over ZMQ PUSH to the Router.
3. Allocate `ReqState(strip_prefix, prev_text="", out_list=[],
   finished=False, error=None, event=asyncio.Event())` and register it in
   `rid_to_state[rid]`.
4. Loop: `await state.event.wait()`; yield the newest frame from
   `out_list`; clear `out_list` and `event`. Break on `state.finished`
   and remove the entry from `rid_to_state`.

**`handle_loop`** — background task draining `BatchStrOut` from the
DetokenizerManager.

1. `recv_obj = await recv_from_detokenizer.recv_pyobj()`.
2. For each `(rid, full_text, finished, error)`:
   - Look up `state = rid_to_state[rid]`; skip if already unregistered.
   - On the finished frame, strip `state.strip_prefix` from `full_text`
     (so the chat preamble doesn't leak into output).
   - Compute `delta = full_text[len(state.prev_text):]`.
   - Append `{"text": full_text, "delta": delta, "finished": finished,
     "error": error}` to `state.out_list`; update `prev_text`; set
     `state.finished = finished`; call `state.event.set()`.

### Router (`serving/router.py`) — the GPU-owning process

Runs in its own process (spawned from `server.py`) on the event loop
started inside `start_router_process`. Two async tasks race:
`loop_for_recv_requests` and `loop_for_forward`.

**Initialization** (inside `start_router_process`)

1. Build the single `LLMEngine`: `LLMEngine.from_engine_args(engine_args)`.
   This is the slow step (~30–90 s).
2. Bind two ZMQ sockets: PULL from TokenizerManager, PUSH to
   DetokenizerManager.
3. Initialize `recv_reqs: List[TokenizedGenerateReqInput] = []`,
   `seq_id_to_rid: Dict[int, str] = {}`, `seq_id_to_input_len: Dict[int, int] = {}`.
4. Signal `"ok"` back to the parent through the `mp.Pipe`, then start both
   tasks.

**`loop_for_recv_requests`** — `await recv_from_tokenizer.recv_pyobj()` in a
tight loop, appending to `self.recv_reqs`. Completely non-blocking WRT the
engine step, so new arrivals don't wait for the current GPU step to finish.

**`loop_for_forward`** — the engine step loop:

1. `_admit_new_requests()`:
   - Move everything from `recv_reqs` to a local list (swap, don't iterate
     while being appended to).
   - For each `TokenizedGenerateReqInput`:
     - `seq_id = llm_engine.add_request(input_ids, sampling_params, ...)`.
     - Record `seq_id_to_rid[seq_id] = rid` and
       `seq_id_to_input_len[seq_id] = len(input_ids)`.
2. `outs = self.llm_engine.step()` — runs one IFB step on the GPU.
   Returns `List[dict]`, one entry per currently-active sequence:
   `{"seq_id": int, "tokens": List[int], "text": Optional[str],
   "finished": bool}`. Note the asymmetry: `tokens` is the entire id
   sequence (prompt + generated so far), `text` is only populated when
   `finished=True` (at which point it's the authoritative full decode
   without `skip_special_tokens`).
3. `_send_step_outputs(outs)` — build a `BatchTokenIDOut`:
   - For each `o`:
     - `seq_id = o["seq_id"]`; translate via `seq_id_to_rid` to get `rid`;
       read `prompt_len = seq_id_to_input_len[seq_id]`.
     - If `o["finished"]`: push `rid`, `output_tokens=[]`,
       `final_text=o["text"]`, `finished=True`, and `pop` both maps.
     - Otherwise: `gen = o["tokens"][prompt_len:]` (cumulative generated
       ids, prompt stripped); push `rid`, `output_tokens=gen`,
       `final_text=None`, `finished=False`.
   - `send_to_detokenizer.send_pyobj(BatchTokenIDOut(...))`.

The `seq_id → rid` mapping matters because `LLMEngine` assigns monotonic
integer ids internally while the outside world (HTTP clients, our
`rid_to_state` dict) uses UUID strings. Without this map the same engine
reuse of small ids would collide across requests.

### LLMEngine step — where continuous batching lives (upstream OmniServe)

`LLMEngine.step()` is OmniServe's in-flight batching primitive. In one
call it does:

1. Schedule which waiting/running sequences run this step (respecting
   `--max-num-seqs`, `--max-num-batched-tokens`, `--chunk-prefill-size`).
2. Run one fused GPU forward: prefill for newly admitted sequences is
   concatenated with a single decode step for already-running ones. This
   is the continuous-batching trick — newly arriving requests don't wait
   for the existing batch to drain.
3. Run the sampler. The greedy path (`torch.argmax`) is taken when
   `temperature < 1e-5 or top_p < 1e-8 or top_k == 1`; otherwise
   `softmax + multinomial`. `ignore_eos=True`, `stop`, and `stop_token_ids`
   from `SamplingParams` are honored here.
4. Check termination; sequences that hit max-tokens / EOS / stop strings
   get `finished=True` and an authoritative full `text` decode in the
   returned dict.

We interact with the engine purely through `add_request`, `step`, and
`abort` — we do not touch its internal scheduler, KV cache, or kernels.

### DetokenizerManager (`serving/detokenize_manager.py`)

Runs in P2. The point of this process is to keep the Router GPU-pure — the
HF tokenizer is pickled Python code, and running it in the Router would
steal CPU cycles from the engine step loop.

**Initialization**
- Load HF `AutoTokenizer` (CPU only, no CUDA).
- Bind ZMQ: PULL from Router, PUSH to TokenizerManager.
- Signal `"ok"` on the parent pipe.

**`handle_loop`**

1. `batch = await recv_from_router.recv_pyobj()` — a `BatchTokenIDOut`.
2. For each `(rid, tokens, finished, final_text)`:
   - If `final_text is not None` (finished request): use it directly —
     the engine already ran `tokenizer.decode(all_ids)` and that decode is
     authoritative (it correctly handles BPE merges and special tokens).
   - Else (intermediate streaming frame): `output_str =
     tokenizer.decode(tokens, skip_special_tokens=False)`. `tokens` here
     is only the generated portion (prompt stripped by the Router), and
     `skip_special_tokens=False` is required so the TokenizerManager's
     `strip_prefix` logic lines up character-for-character.
3. `send_to_tokenizer.send_pyobj(BatchStrOut(rids, output_strs,
   finished))`.

### IPC contracts (`serving/io_struct.py`)

Three pickle-safe dataclasses travel across the ZMQ sockets:

- `TokenizedGenerateReqInput(rid: str, input_ids: List[int],
  sampling_params: SamplingParams, stream: bool)` — TokenizerManager → Router.
- `BatchTokenIDOut(rids: List[str], output_tokens: List[List[int]],
  finished: List[bool], final_text: List[Optional[str]],
  errors: List[Optional[str]])` — Router → DetokenizerManager.
- `BatchStrOut(rids: List[str], output_strs: List[str],
  finished: List[bool], errors: List[Optional[str]])` —
  DetokenizerManager → TokenizerManager.

Batches carry multiple `rid`s because the Router's `step()` can finalize
or advance several sequences in one GPU call.

Out of scope (not implemented): speculative decoding, prefix / radix
cache, multi-GPU TP beyond what the engine already does, tool / function
calling.

---

## Repo layout

The online-serving additions all live at the repo root and under `serving/`;
everything else is the original OmniServe engine (kept intact).

```
serving/                       # NEW – three-process serving stack
├── server.py                  #   entry point – spawns P2, P1, starts FastAPI
├── tokenize_manager.py        #   P0 helper (tokenize + per-request state)
├── router.py                  #   P1 – LLMEngine loop
├── detokenize_manager.py      #   P2 – token-ids -> strings
├── openai_api.py              #   /v1/models, /v1/completions, /v1/chat/completions
├── io_struct.py               #   pickle-safe IPC dataclasses
└── log_utils.py               #   vlog / ilog / elog (SERVING_VERBOSE env var)

bench_serving.py               # NEW – SGLang v0.4.0 benchmark script, unmodified
gradio_ui.py                   # NEW – chat UI over /v1/chat/completions
gradio_completion.py           # NEW – simpler UI over raw /generate

omniserve/                     # original engine (LLMEngine, kernels, modeling…)
kernels/                       # original CUDA kernels
qserve_e2e_generation.py …     # original offline scripts, untouched
```

---

## Quick start

### 1. Start the server

Run from the repo root (this directory):

```bash
export MODEL_PATH=./qserve_checkpoints/Llama-3-8B-Instruct-QServe

NUM_RETRIEVAL_GPU_PAGE_BLOCKS=4000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=0 \
CHUNK_PREFILL_SIZE=65536 \
python -m serving.server \
    --model "$MODEL_PATH" \
    --quant-path "$MODEL_PATH" \
    --precision w4a8kv4 \
    --group-size -1 \
    --ifb-mode \
    --kv-quant-granularity fine_grained \
    --max-num-seqs 8 \
    --max-num-batched-tokens 262144 \
    --chunk-prefill-size 8192 \
    --sparse-decode-mode 0 \
    --served-model-name llama-3-8b-qserve
```

First startup takes ~3–4 min on an 8B W4A8KV4 model (weight load + KV
cache allocation + CUDA kernel warmup). Wait for `[server] listening on
http://127.0.0.1:8000` before sending requests.

`--sparse-decode-mode 0` is **required** with `--kv-quant-granularity
fine_grained`: without it the engine takes the dynamic-sparse decode
kernel path, which has a `tokens_per_block % (K_LOOP_UNROLL * K_PER_ITER)
== 0` constraint that fails on the standard KV-cache config and
immediately asserts at the first decode step.

Set `SERVING_VERBOSE=1` to see per-request IPC traces.

### 2. Hit it with curl

```bash
# native endpoint (not OpenAI-shaped)
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_new_tokens":16,"stream":true}'

# OpenAI chat, streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3-8b-qserve","stream":true,
       "messages":[{"role":"user","content":"Write a haiku."}]}'
```

### 3. Use the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="llama-3-8b-qserve",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=16,
)
print(resp.choices[0].message.content)   # "Paris."
```

### 4. Chat in a browser

```bash
python gradio_ui.py        # http://127.0.0.1:7860
```

---

## Benchmarking with SGLang's official script

We deliberately don't ship a custom benchmarker — we target the OpenAI API
contract that SGLang's [`bench_serving.py`][bench] expects, so their script
works out of the box.

[bench]: https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py

A copy of v0.4.0's self-contained version lives at `bench_serving.py` in
this repo. Dependencies: `aiohttp numpy tqdm requests transformers`.

### One-off run

```bash
python bench_serving.py \
    --backend vllm \
    --host 127.0.0.1 --port 8000 \
    --model llama-3-8b-qserve \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --num-prompts 20 --max-concurrency 4 \
    --random-input-len 256 --random-output-len 64 \
    --disable-ignore-eos
```

`--backend vllm` selects the OpenAI-completions client path (which is the
API shape we expose). Sample output:

```
============ Serving Benchmark Result ============
Backend:                                 vllm
Max reqeuest concurrency:                4
Successful requests:                     20
Benchmark duration (s):                  2.26
Total input tokens:                      2897
Total generated tokens:                  541
Request throughput (req/s):              8.86
Input token throughput (tok/s):          1282.87
Output token throughput (tok/s):         239.57
Total token throughput (tok/s):          1522.44
Mean E2E Latency (ms):                   390.77
Mean TTFT (ms):                           32.61
P99 TTFT (ms):                            74.24
Median TPOT (ms):                         11.18
Mean ITL (ms):                            11.04
==================================================
```

### Flags worth knowing

| flag                       | notes                                                |
|----------------------------|------------------------------------------------------|
| `--disable-ignore-eos`     | recommended. Without it the bench asks the server to generate exactly `--random-output-len` tokens by ignoring EOS. The `ignore_eos=True` code path through the OmniServe sampler has not been fully validated on this kernel build; pass this flag and let natural EOS terminate — output length distributions still come from the dataset. |
| `--disable-stream`         | send `stream: false` in requests (useful to isolate server-side cost from SSE framing). |
| `--dataset-name sharegpt`  | more realistic length distributions. Needs `--dataset-path` pointing at a `ShareGPT_V3_unfiltered_cleaned_split.json`. |
| `--request-rate 8`         | Poisson arrivals at 8 req/s instead of back-to-back. |
| `--backend sglang`         | triggers SGLang-native extensions including a warmup `POST /flush_cache`. We implement that endpoint as a no-op so this backend also works; `vllm` backend is simpler. |

### Concurrency sweep results

Run on Llama-3-8B-Instruct W4A8KV4 with `--max-num-seqs 8`, 300 random-length
prompts at `--random-input-len 256 --random-output-len 128`, Poisson arrivals
at rates {4, 8, 12, 16, 20, 24, 28} req/s (`--multi --request-rate-range
4,32,4`):

```bash
python bench_serving.py \
    --backend vllm --host 127.0.0.1 --port 8000 \
    --model llama-3-8b-qserve --tokenizer "$MODEL_PATH" \
    --dataset-name random --num-prompts 300 \
    --random-input-len 256 --random-output-len 128 \
    --multi --request-rate-range 4,32,4 \
    --disable-ignore-eos
```

| Rate (req/s) | Duration (s) | Req tput (req/s) | Out tput (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean ITL (ms) | Mean E2E (ms) |
|---|---|---|---|---|---|---|---|
| 4  | 75.8 | 3.96  | 260.9 | 30.8   | 53.2    | 10.8 | 735.9  |
| 8  | 36.5 | 8.21  | 541.2 | 101.4  | 511.6   | 11.6 | 860.3  |
| 12 | 30.1 | 9.97  | 656.9 | 1787.4 | 4164.6  | 11.7 | 2553.1 |
| 16 | 30.2 | 9.94  | 655.4 | 4839.1 | 9431.8  | 11.9 | 5614.9 |
| 20 | 30.1 | 9.96  | 656.7 | 6570.6 | 13970.2 | 12.0 | 7348.4 |
| 24 | 29.7 | 10.12 | 666.9 | 7480.7 | 15077.8 | 11.8 | 8248.0 |
| 28 | 29.6 | 10.14 | 668.4 | 8240.8 | 16776.8 | 11.7 | 9004.3 |

<p align="center">
  <img src="./docs/benchmark/throughput.png" width="720" alt="Throughput saturates around 10 req/s"><br>
  <img src="./docs/benchmark/ttft.png" width="720" alt="TTFT explodes past capacity (log scale)"><br>
  <img src="./docs/benchmark/tpot.png" width="720" alt="TPOT stays flat across the sweep">
</p>

A few things stand out:

- **Request throughput saturates around ~10 req/s (~660 tok/s output).**
  Between 12 and 28 req/s offered load, measured throughput is flat — the
  engine is capped at `--max-num-seqs 8` for this I/O length mix. Raising
  `--max-num-seqs` and KV page blocks would push the ceiling higher, at the
  cost of per-step latency.
- **TTFT explodes the moment offered rate exceeds capacity.** 30 ms at
  4 req/s, 100 ms at 8 req/s, 1.8 s at 12 req/s, 8.2 s at 28 req/s — a
  textbook queueing curve. P99 TTFT at 28 req/s is 16.8 s: requests sit in
  the queue that long before their first token.
- **ITL is rock-steady at ~10–12 ms across the whole sweep.** Per-sequence
  decode throughput is unaffected by queue depth, which is exactly what
  continuous batching is supposed to give you: once a request is in the
  running batch, latecomers don't slow it down.
- **E2E latency is TTFT-dominated under load.** At low rate, E2E ≈ 128 × ITL
  ≈ 1.3 s (the 735 ms mean at rate 4 is lower because not every prompt runs
  all 128 output tokens). Above saturation, E2E tracks TTFT almost 1:1 —
  time-in-queue is the cost.

---

## Environment variables

| name                 | default  | meaning                                          |
|----------------------|----------|--------------------------------------------------|
| `SERVING_VERBOSE`    | `0`      | `1` enables per-request vlog traces in every proc |
| `SERVER_URL`         | (see ui) | base URL for `gradio_ui.py`                      |
| `MODEL_NAME`         | llama-3-8b-qserve | model name used by `gradio_ui.py`        |
| `GRADIO_HOST/PORT`   | 127.0.0.1 / 7860 | UI bind address                          |

Key CLI flags added on top of `LLMEngine`'s own:

| flag                    | meaning                                                    |
|-------------------------|------------------------------------------------------------|
| `--host / --port`       | FastAPI bind                                               |
| `--router-port`         | ZMQ port TokenizerManager -> Router (default 8500)         |
| `--detokenizer-port`    | ZMQ port Router -> Detokenizer (default 8510)              |
| `--tokenizer-port`      | ZMQ port Detokenizer -> TokenizerManager (default 8520)    |
| `--served-model-name`   | name returned by `/v1/models` and accepted by `/v1/...`    |

---

## License

Apache 2.0, same as upstream [OmniServe](https://github.com/mit-han-lab/omniserve)
and [SGLang](https://github.com/sgl-project/sglang). See `LICENSE`.

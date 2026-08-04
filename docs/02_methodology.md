# 02 · Methodology

> 中文速读：讲清楚"怎么比"——同一份导出的 Qwen2 主干、同一份负载（prefill 64 + 生成 48 token）、同一张 H100、bf16、warmup 后计时；以及公平性边界。

## Goal

Measure the **LLM-backbone** inference speed of eager PyTorch vs vLLM vs TensorRT-LLM on identical hardware, dtype, and workload, then place it in the context of full end-to-end TTS latency.

## Model under test

- **Source**: `rednote-hilab/dots.tts-base` (Hugging Face), a multi-stage TTS.
- **Backbone**: the internal `Qwen2ForCausalLM` — 1.543B params, 28 layers, `hidden_size=1536`, 12 attention heads / 2 KV heads (GQA), `intermediate=8960`, `vocab=151,672`, `tie_word_embeddings=True`, RoPE θ=1e6.
- **Export**: `scripts/export_backbone.py` strips the `llm.` prefix from the TTS checkpoint's `model.safetensors` (338 tensors) and writes a standalone HF `Qwen2ForCausalLM` directory (verified to load via `AutoModelForCausalLM`, 1.543B, bf16). Tokenizer files are copied from the source.
  - Why export? vLLM and TensorRT-LLM are LLM engines; the full TTS `DotsTtsCore` (LLM + DiT + encoders) is not servable by either. Isolating the backbone makes the three engines comparable.
- **Verified separately**: the full TTS pipeline does synthesize correctly on this server, so the exported backbone is the real one, not a stub.

## Workload (identical across engines)

- **Prefill**: a 64-token prompt (synthetic token ids `10..73`; the numeric values are arbitrary — the backend cost is sequence length, not semantics).
- **Generation**: 48 autoregressive tokens (temperature 0).
- **Precision**: `bfloat16`.
- **Batch**: 1 (single request). Chosen because TTS is a latency-sensitive, low-concurrency workload.

> Rationale for P=64/G=48: mirrors the shape of a real dots-tts synthesis (the audited pipeline decodes ~48 audio patches). Synthetic tokens avoid tokenizer/BPE variance; both engines receive the same integer ids via `prompt_token_ids`.

## Measurement

- Each engine is **warmed up once** before timing (cold-launch/compile costs excluded).
- Three timed runs; **best-of-3** reported (and all raw runs kept in `results/*.json`).
- Metric per engine: end-to-end wall time for prefill+48-token decode, per-token decode ms, tokens/s.
- Engines:
  - `bench_eager.py`: `AutoModelForCausalLM` + `transformers.StaticCache`, one token per decode step (mirrors how dots-tts drives its LLM).
  - `bench_vllm.py`: `vllm.LLM(model=backbone, dtype=bfloat16)` default optimized config.
  - `trtllm_bench.py`: `tensorrt_llm.LLM(model=backbone, dtype=bfloat16)` — **PyTorch backend** (the 1.3.0 default). Inputs passed as `TokensPrompt`.

## Environment

- GPU: NVIDIA H100 NVL, 94 GiB, driver 580.82.07, CUDA 13.0 (host). All runs pinned to one physical GPU.
- Software:
  - eager: torch 2.12.1+cu130, transformers 5.12.1 / 4.57.x
  - vLLM: 0.17.1 (conda env `vllm-qwen`, torch 2.10.0)
  - TRT-LLM: NGC `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23` (torch 2.12.0a0+cu13.2)

## Fairness caveats (read before quoting)

1. **TRT-LLM numbers are PyTorch-backend** — the current default execution stack of TRT-LLM 1.3.0, *not* a compiled TensorRT engine (the builder was removed in 1.3; see README). If the intent is "compiled-TRT-engine", this comparison under-states what a hand-built TRT engine could reach; that path needs the 1.2.x toolchain.
2. **Batch=1, small model (1.5B)** — latency/overhead-bound. Both vLLM and TRT-LLM are tuned for larger models / higher concurrency; relative rankings can shift at different scale. A 1.5B single-stream request is far outside the sweet spot of either engine.
3. **Best-of-3, single node, single sample sentence-length.** Not a statistical sweep. Run-to-run variance is low (see raw runs).
4. The end-to-end dots-tts number is the **eager integration** (its built-in profiler), not served by any engine.

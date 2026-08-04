# vLLM vs TensorRT-LLM on a TTS Model's LLM Backbone

A reproducible benchmark comparing three inference paths for the **autoregressive LLM backbone of a modern multi-stage TTS model** on NVIDIA H100:

- **eager PyTorch** (baseline)
- **vLLM 0.17.1**
- **TensorRT-LLM 1.3.0** (PyTorch backend — see [the honest caveat below](#what-exactly-is-tensorrt-llm-here))

> 中文速读：本仓库用一套可控负载，在 **dots-tts（1.543B Qwen2 主干，28 层）** 上对比 eager PyTorch / vLLM / TensorRT-LLM 三种推理路径，并给出**端到端 TTS 延迟拆分**（扩散/LLM/vocoder 各占多少）。核心结论：
> - 在 LLM 主干一层，**vLLM ≈ 453 tok/s > TensorRT-LLM ≈ 213 tok/s >> eager ≈ 21 tok/s**（vLLM 约快 TRT-LLM 2×，两者约快 eager 10–35×）。
> - 但**整条 TTS 里 LLM 只占 ~21% 延迟**，扩散流匹配层占 ~68%——而后者的优化**不在 vLLM/TensorRT-LLM 的定义域内**。引擎优化对端到端 TTS 的杠杆有限（~20%）。
> - 诚实边界：TensorRT-LLM 1.3.0 **移除了经典"编译 TensorRT 引擎"路径**，默认 PyTorch 执行栈；真·编译 TRT 引擎需 1.2.x 旧工具链。

---

## TL;DR (results)

Same workload across all three engines: **prefill 64 tokens + 48 autoregressive tokens**, bytes/s measured post-warmup, **bf16, single H100 NVL, batch=1**.

| Engine | total (s) | per-token (ms) | tokens/s | vs eager |
|---|---|---|---|---|
| eager PyTorch (StaticCache) | 3.70 | 46.7 | 21.4 | 1.0× |
| **TensorRT-LLM 1.3.0** (PyTorch backend) | 0.226 | 4.71 | 212.5 | **9.9×** |
| **vLLM 0.17.1** | **0.106** | **2.21** | **452.8** | **34.9×** |

> Runner-up engine. vLLM is ~2.1× faster than TensorRT-LLM on this small backbone at batch=1; both are an order of magnitude faster than eager.

### End-to-end context (why the LLM layer is the wrong bottleneck)

End-to-end synthesis of a 7.7 s utterance in dots-tts (full pipeline, eager, bf16, 1×H100) breaks down as:

| Stage | sec | share |
|---|---|---|
| **Flow-matching DiT (diffusion/ODE)** | 18.56 | **67.8%** |
| **LLM backbone (Qwen2 AR decode)** | 5.69 | **20.8%** |
| patch encoder / latent decoder / other | 2.31 | 8.5% |
| (speaker / vocoder / prefill / misc) | ~0.8 | ~3% |
| **Total** | **27.36** | 100% (RTF 3.56) |

Because the LLM layer is only ~21% of end-to-end latency, replacing it with vLLM (≈ 0.1–0.2 s) cuts the total from ~27.4 s to ~21.8 s — **a ~20% end-to-end win**, not the order-of-magnitude the engine-level numbers suggest. The diffusion stage is where TTS time actually goes, and **neither vLLM nor TensorRT-LLM serves a diffusion stage**.

---

## What exactly is "TensorRT-LLM" here?

A finding, not an assumption: **TensorRT-LLM 1.3.0 (release:1.3.0rc23) no longer ships the classic "compile a TensorRT engine" path** — there is no `trtllm-build`, no `convert_checkpoint`, no `tensorrt_llm.builder` module. The CLI exposes `trtllm-serve`, `trtllm-bench`, `trtllm-eval`, and the high-level `LLM` API now defaults to a **PyTorch execution backend** (optimized torch, FlashAttention, graph compilation) with an `_autodeploy` edge alternative.

So "TensorRT-LLM vs vLLM" in this repo is measured on **TensorRT-LLM's current default execution stack**, which is a legitimate, supported TensorRT-LLM path — but it is **not a compiled TensorRT engine**. If you specifically want the classical compiled-TRT engine number, see `docs/05_reproduction.md` (requires the older 1.2.x toolchain, which still bundles TensorRT 10.14 and the removed builder).

> 中文速读：所谓"TensorRT-LLM vs vLLM"这里= TRT-LLM **当前默认（PyTorch backend）**执行栈。1.3.0 已下线经典编译 TRT 引擎路径。若坚持要"编译后 TRT 引擎"的数，需回退 1.2.x（见复现文档）。

---

## Model under test

- **dots-tts** (`rednote-hilab/dots.tts-base`), a multi-stage TTS with a **1.543B, 28-layer `Qwen2ForCausalLM`** backbone (hidden 1536, 12 heads / 2 KV heads, vocab 151,672, weight-tied).
- The benchmark targets **only the exported standalone Qwen2 backbone** (`llm.*` weights, see `scripts/export_backbone.py`) so that vLLM and TensorRT-LLM — both LLM engines — compete on an apples-to-apples unit.
- Diffusion DiT, vocoder, and encoders are *not* part of the engine comparison (neither engine serves them); they appear only in the end-to-end breakdown.

## Why multi-stage TTS is the interesting case

TTS models come in two broad families. This benchmark is specifically about the **diffusion/multi-stage family**:

- **Multi-stage / diffusion TTS** (dots-tts, VoxCPM2): LLM predicts latents, then a **diffusion flow-matching** stage refines them, then a vocoder renders audio. The LLM is a fraction of the cost; the diffusion stage dominates.
- **Pure-AR TTS**: the LLM directly emits waveform patches (little/no diffusion). Here the LLM layer *is* the whole model, so vLLM/TensorRT-LLM have the full leverage.

The insight generalizes: **the value of an LLM serving engine scales with the LLM's share of end-to-end cost.**

---

## Reproduce

Full steps (conda/venv, container, commands) in [`docs/05_reproduction.md`](docs/05_reproduction.md). One-line summary:

```bash
# 0. export the Qwen2 backbone from the TTS checkpoint (once)
python scripts/export_backbone.py          # -> backbone_qwen2/

# 1. eager baseline
CUDA_VISIBLE_DEVICES=1 python scripts/bench_eager.py 64 48

# 2. vLLM (conda env vllm-qwen, vllm 0.17.1)
CUDA_VISIBLE_DEVICES=1 python scripts/bench_vllm.py 64 48

# 3. TensorRT-LLM (NGC container nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23)
docker run --gpus all ... tensorrt-llm/release:1.3.0rc23 python /bench/trtllm_bench.py
```

Raw data in [`results/`](results/).

---

## Docs

- [`docs/01_background.md`](docs/01_background.md) — why compare; TTS architecture; vLLM vs TensorRT-LLM positioning
- [`docs/02_methodology.md`](docs/02_methodology.md) — experiment design, fairness, environment
- [`docs/03_results.md`](docs/03_results.md) — full tables + end-to-end breakdown
- [`docs/04_analysis.md`](docs/04_analysis.md) — interpretation, caveats, production guidance
- [`docs/05_reproduction.md`](docs/05_reproduction.md) — exact steps on 2×H100

## License

MIT. Model/engine licenses are NVIDIA / the model owners' own (PLE/TRT-LLM as shipped).

---
*Benchmark artifacts and numbers in this repo were produced by running the scripts here on a 2× H100 NVL server.*

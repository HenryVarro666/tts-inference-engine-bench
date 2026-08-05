# 04 · Analysis

> 中文速读：怎么解读这些数——为什么 vLLM 在这快、为什么 TTS 瓶颈在扩散层、engine 对 TTS 到底有多大杠杆，以及给生产的建议与诚实边界。

## 4.1 Why vLLM beats TRT-LLM here (and both beat eager)

- **eager → engine (×10–×35)**: dominated by kernel launch / Python interpreter / lack of graph capture. `transformers` eager does a per-token Python forward; vLLM and TRT-LLM capture the decode step (`CUDA graphs` / compiled graphs) so each token re-instantiates kernels instead of re-launching + re-interpreting.
- **vLLM → TRT-LLM (×2)**: at **batch=1 on a 1.5B model**, both engines are overhead-bound and neither is in its sweet spot; vLLM's smaller/leaner per-request path wins. This ordering is **not general** — at higher batch / larger token-width, TRT-LLM's kernel tuning can close or invert it. The honest statement is *"within measurement error of each other's regime, vLLM edges ahead at this tiny scale"*, not a universal ranking.

## 4.2 Why TTS latency is a diffusion problem, not an LM problem

The dots-tts split puts **67.8% of end-to-end time in flow-matching (DiT+ODE)**, and only **20.8% in the LLM**. Consequences:

1. Replacing eager LLM with vLLM/TRT-LLM buys at most **~20%** end-to-end (i.e., the LLM's share), because the diffusion stage is untouched.
2. To actually speed up a diffusion TTS you must optimize the **DiT / ODE integrator** — batch the ODE over patches, fuse the UNet/DiT with TRT/ONNX/TensorRT, reduce `num_steps` (e.g. distillation), or raise the flow-matching efficiency. That is a *different* axis than LLM serving engines.
3. For **pure-AR TTS** (LLM ≈ 100% of cost), the LLM engine *is* the whole lever — the opposite end of the spectrum.

## 4.3 The honest boundary on "TensorRT vs vLLM"

TRT-LLM 1.3.0 removed the classical compiled-TRT-engine path (no `trtllm-build`/`builder`); its shipped default is a PyTorch execution backend. So:

- These numbers are a fair **"engine A vs engine B current-stack"** comparison — both are the current shipping products.
- They are **not** a "hand-compiled TensorRT graph vs vLLM" comparison. A hand-built TRT engine for the Qwen2 backbone (via 1.2.x toolchain) could be faster than what TRT-LLM's default stack shows; but that path was removed from modern TRT-LLM, which is itself a meaningful product finding.

## 4.4 Production guidance

- If you serve the **LLM stage of a diffusion TTS**: use **vLLM** — fastest, simplest HF-compatible, production-proven.
- If you need **TensorRT** on the fleet (NVIDIA-only stack, strict TensorRT deployment): budget a **1.2.x classic-engine build** for the compiled path, or accept TRT-LLM's PyTorch backend as the supported default.
- To cut **end-to-end** diffusion-TTS latency, do not stop at the LLM engine — spend effort on the **flow-matching / vocoder** stages (`num_steps`, distillation, DiT kernel fusion). That is where 2/3 of the time lives.
- Always benchmark at **your real batch and model size**; these results are batch=1 on 1.5B and may not extrapolate to 7B+ / high concurrency.

## 4.5 What this means for a speech-data/AI-data role (why it's worth having measured)

When scoping latency/cost for AI-speech products: *don't* assume LLM-engine work solves speech latency. Speak to the **stage-ownership** of the latency budget — the diffusion stage owns it here. A data/AI engineer who can produce a measured latency split (LLM vs diffusion vs vocoder) and reason about where engine vs. model-level (step-distillation, data for shorter utterances) levers land, has a concrete, evidence-backed view of the system.

## 5. Key lessons (transferable)

> 中文速读：① 基准粒度必须匹配架构（组件胜利≠系统胜利）；② 先测延迟预算再谈优化；③ 工具栈流动、命名有歧义，引用基准要钉死版本/backend/硬件/batch/精度；④ 引擎杠杆 ∝ LLM 占端到端成本比例，可一般化。

1. **Benchmark granularity must match the architecture — a component win is not a system win.** "vLLM beats TRT-LLM 2×" holds only for the LLM layer; because diffusion owns ~68% of the TTS pipeline, the end-to-end gain is only ~20%. Always isolate the architectural unit before declaring a winner.
2. **Measure the latency budget before optimizing.** The end-to-end split is the highest-value output of this exercise — it refutes the intuitive "faster LLM engine → faster TTS". The right decision input is *where the time goes*, not *which engine is fastest*.
3. **Tooling stacks are fluid and names are ambiguous.** "TensorRT" (a graph compiler) and "TensorRT-LLM" (a framework — whose 1.3 default is now a PyTorch backend, with the compiled-engine path removed) are different things. When you quote any benchmark, pin version + backend + hardware + batch + precision, or it stops being true the moment the product changes stack.
4. **The leverage generalizes.** `value of an LLM serving engine ∝ the LLM's share of end-to-end cost` — ≈100% for pure-AR TTS, ≈21% for diffusion TTS. This is a reusable framework for deciding where to invest across compound models.

## 6. Applying in industry

> 中文速读：六条落地建议 —— ① 延迟预算/阶段归属立为工程惯例；② 按阶段选引擎不按品牌；③ 扩散受限时模型级杠杆>引擎级；④ 实时口译目标 e2e RTF<1，主攻扩散+流式；⑤ 生产基准纪律（保留 latency-budget 文档当唯一权威）；⑥ 数据岗位的间接杠杆（数据→RTF 回归评测）。

1. **Make latency-budget / stage-ownership profiling a standard practice.** For any multi-stage or compound model, generate a per-stage RTF table (profiler / nsys / CUDA events) *before* choosing serving infrastructure. Report "where the time is spent" rather than "which engine is fastest".

2. **Choose engines per stage, not by brand.** In a multi-stage speech stack: an LLM serving engine (vLLM / TRT-LLM) for the autoregressive stage, plus a *dedicated* path for the DiT / vocoder (ONNX Runtime, TensorRT, torch.compile, Triton). The highest-leverage diffusion optimization is the **ODE**: fewer `num_steps` via distillation, and batching the ODE across patches.

3. **When diffusion-bound, model-level levers beat engine-level ones.** Reduce ODE steps (distillation), use a cheaper/faster vocoder, enable **streaming / chunked synthesis** to cut time-to-first-audio (TTFA), and lean on data-side levers (shorter, cleaner utterances).

4. **For real-time speech / interpretation, the target is e2e RTF < 1 and bounded TTFA.** Here the LLM stage is already RTF 0.74 even in eager (faster than realtime); the *wall* is the diffusion stage at RTF 2.4. To reach near-realtime, attack the diffusion/acoustic stage and streaming — not another LLM server.

5. **Production benchmark discipline.** Pin version + backend + GPU + batch + dtype; keep raw runs; re-benchmark at your *real* concurrency and model size; re-measure whenever the stack changes. Maintain a single **latency-budget document** as the source of truth rather than a headline number.

6. **Data teams have real but indirect leverage on latency.** Data shapes it via shorter/cleaner utterances, speaker-disjoint splits for acoustic training, and **RTF regression tests on a fixed eval set**. A measured latency split is the *hard evidence* for arguing "invest in diffusion + evaluation data, not another LLM engine" — far more persuasive than an unquantified optimization claim.

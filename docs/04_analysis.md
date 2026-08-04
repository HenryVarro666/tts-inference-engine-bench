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

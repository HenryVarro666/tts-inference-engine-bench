# 01 · Background

> 中文速读：这一篇讲清楚"为什么在 TTS 上比 vLLM vs TensorRT-LLM"——TTS 分两大类，本仓库只讨论 diffusion/多阶段这一类；以及为什么这两个引擎只能比"LLM 主干"这一层。

## Why benchmark inference engines on a TTS model

TTS (text-to-speech) latency is a first-order product and cost concern, and the serving-engine landscape (vLLM, TensorRT-LLM, TensorRT, etc.) is mature for *language models*. A natural question is: **how much of that engine optimization transfers to speech models?**

The answer is *not obvious* because most modern high-quality TTS models are **not a single language model**. They compose several stages, and LLM serving engines are only meaningful for the autoregressive stage.

## The two TTS families

**A. Multi-stage / diffusion TTS** (`dots-tts`, `VoxCPM2`, most expressive systems)
```
text/semantic  ->  LLM backbone (AR, predicts latents)  ->  diffusion flow-matching (DiT, ODE)  ->  vocoder
```
The LLM is one component. The diffusion stage (many ODE steps) and the vocoder often dominate latency.

**B. Pure-autoregressive TTS** (a minority, e.g. some tokenizer-free generators)
```
text -> LLM (AR, directly emits waveform-patch latents, minimal/no diffusion)
```
Here the LLM *is* the model, so LLM-engine optimization has the full leverage.

This benchmark intentionally studies **family A** — the harder, more realistic case where the answer to "how much can vLLM/TensorRT-LLM help a TTS?" is *"only as much as the LLM share"*.

## Why only the LLM backbone is compared head-to-head

- **vLLM** is an autoregressive decoder engine. It does not serve diffusion DiTs, vocoders, or multi-stage encoders.
- **TensorRT-LLM** is the same category (LLM-focused).
- Therefore the only apples-to-apples unit for "vLLM vs TensorRT-LLM on a TTS" is the **TTS model's LLM backbone**, isolated and exported as a standalone `Qwen2ForCausalLM`.

The diffusion/vocoder stages are still *measured* (in the end-to-end breakdown) but compared on a different axis (eager vs. what a graph compiler could do), which neither vLLM nor TRT-LLM provides.

## Positioning of the engines

| Engine | Kind | Serves | Notes |
|---|---|---|---|
| eager PyTorch | framework runtime | anything (slow for LLM AR) | the raw baseline |
| vLLM 0.17.1 | LLM serving engine | AR decoder (HF arch) | CUDA graphs + torch.compile + optimized kernels + PagedAttention |
| TensorRT-LLM 1.3.0 | LLM serving engine | AR decoder | **current default = PyTorch backend**; classic TRT-engine path removed in 1.3 (see [README](../README.md)) |

# Benchmark problem

> **Under identical model, workload and timing definitions, how much does changing the autoregressive serving engine improve complete TTS latency rather than only backbone token throughput?**

**Why it matters:** An isolated component speedup can overstate the benefit users hear. Engine selection should follow measured end-to-end costs and quality constraints.

**Central tension:** Highly optimized decoding may accelerate one stage while acoustic generation, initialization or batching remains dominant. Warmup, token-count denominators and aggregation choices can also create apparent engine differences.

**Current evidence and gap:** [README.md](README.md) describes eager, vLLM and TensorRT-LLM comparisons and a TTS latency breakdown. Historical component figures must be interpreted under their original measurement conditions; they do not establish every backend's end-to-end gain or a current software ranking.

**Decisive verification:** Freeze weights, precision, prompt/output lengths, seeds and hardware. Use the same warmup and repeated-run protocol for every engine, retain all trials and failures, and report consistent throughput denominators and latency distributions. Separately replay identical texts through a fully integrated synthesis path with audio-quality checks. If backbone gains fail to improve complete synthesis, report the bottleneck and limit the engine recommendation accordingly.

This module defines the comparison to verify; it does not rerun or endorse historical numbers.

# vLLM vs TensorRT-LLM on a TTS Model's LLM Backbone

**中文标题：在 TTS 模型 LLM 主干上对比 vLLM / TensorRT-LLM / eager 的推理性能**

A reproducible benchmark comparing three inference paths for the **autoregressive LLM backbone of a modern multi-stage TTS model** on NVIDIA H100: **eager PyTorch**, **vLLM 0.17.1**, and **TensorRT-LLM 1.3.0**.

> 中文速读（TL;DR）：
> 本仓库用一套可控负载，在 dots-tts（1.543B Qwen2 主干，28 层）上对比三种 LLM 推理路径，并给出**端到端 TTS 延迟拆分**。核心结论：
> - LLM 主干一层：**vLLM ≈ 453 tok/s > TensorRT-LLM ≈ 213 tok/s >> eager ≈ 21 tok/s**（vLLM 约快 TRT-LLM 2×，两者约快 eager 10–21×）。
> - 但整条 TTS 里 **LLM 只占 ~21% 延迟**，扩散流匹配层占 ~68% —— 后者不在 vLLM/TensorRT-LLM 的定义域内。引擎优化对端到端 TTS 的杠杆有限（~20%）。
> - 诚实边界：TensorRT-LLM 1.3.0 已**移除经典"编译 TensorRT 引擎"路径**（无 `trtllm-build`），默认跑 PyTorch 执行栈；真·编译 TRT 引擎需回退 1.2.x 旧工具链。

---

## 1. Results 结果

Same workload across all three engines: **prefill 64 tokens + 48 autoregressive tokens**, measured post-warmup, **bf16, single H100 NVL, batch=1**. 三引擎同负载（prefill 64 + 自回归 48 token，warmup 后计时，bf16，单卡 H100 NVL，batch=1）。

| Engine 引擎 | total (s) | per-token (ms) | tokens/s | vs eager |
|---|---|---|---|---|
| eager PyTorch (StaticCache) 基线 | 3.698 | 46.7 | 21.4 | 1.0× |
| **TensorRT-LLM 1.3.0** (PyTorch backend) | 0.226 | 4.71 | 212.5 | **9.9×** |
| **vLLM 0.17.1** | **0.106** | **2.21** | **452.8** | **21.2×** |

> vLLM is ~2.1× faster than TensorRT-LLM on this small (1.5B) backbone at batch=1; both are an order of magnitude faster than eager.
> 在 batch=1 的 1.5B 小模型上 vLLM 约快 TRT-LLM 2.1×；两者都比 eager 快一个数量级。

### End-to-end context 端到端延迟拆分（为什么 LLM 层不是瓶颈）

dots-tts 完整流水线合成 7.68 s 语音（eager，bf16，1×H100）各阶段占比：

| Stage 阶段 | sec | share 占比 |
|---|---|---|
| **Flow-matching DiT (diffusion/ODE) 扩散流匹配** | 18.56 | **67.8%** |
| **LLM backbone (Qwen2 AR decode) 主干** | 5.69 | **20.8%** |
| patch encoder / latent decoder | 2.31 | 8.5% |
| speaker / vocoder / prefill / misc | ~0.8 | ~3% |
| **Total 总计** | **27.36** | 100% (RTF 3.56) |

Because the LLM layer is only ~21% of end-to-end latency, switching it to vLLM (≈ 0.1–0.2 s) cuts the total from ~27.4 s to ~21.8 s — **a ~20% end-to-end win**. The diffusion stage is where TTS time actually goes, and **neither vLLM nor TensorRT-LLM serves a diffusion stage**.

> 因为 LLM 层只占 ~21%，换成 vLLM 最多把端到端从 ~27.4s 降到 ~21.8s（~20% 提升）。TTS 真正的时间在扩散层，而 vLLM / TensorRT-LLM 都不跑扩散。

---

## 2. What exactly is "TensorRT-LLM" here? 这里说的"TensorRT-LLM"是什么

A finding, not an assumption: **TensorRT-LLM 1.3.0 (release:1.3.0rc23) no longer ships the classic "compile a TensorRT engine" path** — there is no `trtllm-build`, no `convert_checkpoint`, no `tensorrt_llm.builder` module. The CLI exposes `trtllm-serve` / `trtllm-bench` / `trtllm-eval`, and the high-level `LLM` API now defaults to a **PyTorch execution backend** (optimized torch + FlashAttention + graph compilation) with an `_autodeploy` edge alternative.

> 这本身是一个发现：TensorRT-LLM 1.3 已下线经典"编译 TRT 引擎"路径（无 `trtllm-build`/`builder`），默认执行栈是 PyTorch backend。因此本仓库的"TensorRT-LLM vs vLLM"是在**当前官方默认执行栈**上的对比，**不是**"手写编译 TRT 图 vs vLLM"。若坚持要编译 TRT 引擎的数，需用仍捆绑 TensorRT 10.14 与旧 builder 的 1.2.x 工具链（详见 [docs/05_reproduction.md](docs/05_reproduction.md)）。

---

## 3. Model under test 被测模型

- **dots-tts** (`rednote-hilab/dots.tts-base`), a multi-stage TTS with a **1.543B, 28-layer `Qwen2ForCausalLM`** backbone (hidden 1536, 12 heads / 2 KV heads, vocab 151,672, weight-tied).
  dots-tts 是一个多阶段 TTS，其内部 LLM 主干为标准 **Qwen2ForCausalLM**（1.543B / 28 层 / 151,672 vocab）。
- The benchmark targets **only the exported standalone Qwen2 backbone** (`llm.*` weights, see `scripts/export_backbone.py`), so the two LLM engines compete on an apples-to-apples unit.
  对比只针对导出的**独立 Qwen2 主干**（见 `scripts/export_backbone.py`），保证 vLLM 与 TRT-LLM 在同一个可比单元上竞争。
- Diffusion DiT / vocoder / encoders are measured only in the end-to-end split (neither engine serves them).
  扩散 DiT / vocoder / 编码器只出现在端到端拆分里（两个引擎都不跑它们）。

### Why multi-stage TTS is the interesting case 为什么多阶段 TTS 是有意思的case

TTS 分两大类，本仓库研究 diffusion/多阶段这一类：

- **Multi-stage / diffusion TTS**（dots-tts, VoxCPM2）: LLM 预测 latent → **diffusion flow-matching** 精修 → vocoder 渲染。LLM 只占一部分开销，扩散层主导。
- **Pure-AR TTS**: LLM 直接自回归输出波形（几乎无扩散）。此时 LLM 层就是整个模型，vLLM/TRT-LLM 有全部杠杆。

**The insight generalizes: the value of an LLM serving engine scales with the LLM's share of end-to-end cost.**（普适洞察：LLM 引擎的价值与 LLM 占端到端成本的比重成正比。）

---

## 4. Reproduce 复现

Full steps (conda/venv, container, commands) in [`docs/05_reproduction.md`](docs/05_reproduction.md). 完整步骤见 `docs/05_reproduction.md`。一键概览：

```bash
# 0. export the Qwen2 backbone from the TTS checkpoint (once)
python scripts/export_backbone.py --src <tts_snapshot> --tgt backbone_qwen2

# 1. eager baseline
CUDA_VISIBLE_DEVICES=1 python scripts/bench_eager.py --model-dir backbone_qwen2

# 2. vLLM (conda env vllm-qwen, vllm 0.17.1)
CUDA_VISIBLE_DEVICES=1 python scripts/bench_vllm.py --model-dir backbone_qwen2

# 3. TensorRT-LLM (NGC container nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23)
docker run --gpus all -v "$PWD":/bench ... python /bench/scripts/trtllm_bench.py --model-dir /bench/backbone_qwen2
```

Raw data in [`results/`](results/). 原始数据在 `results/`。

---

## 5. Docs 文档

- [`docs/01_background.md`](docs/01_background.md) — why compare; TTS architecture; engine positioning 动机 / TTS 架构 / 引擎定位
- [`docs/02_methodology.md`](docs/02_methodology.md) — experiment design, fairness, environment 实验设计 / 公平性 / 环境
- [`docs/03_results.md`](docs/03_results.md) — full tables + end-to-end breakdown 全量表 + 端到端拆分
- [`docs/04_analysis.md`](docs/04_analysis.md) — interpretation, caveats, guidance 解读 / 边界 / 生产建议
- [`docs/05_reproduction.md`](docs/05_reproduction.md) — exact steps + implementation gotchas 复现步骤 + 踩坑记录

## 6. License 许可

MIT. Model/engine licenses are the model owners' own (dot-tts / NVIDIA TRT-LLM as shipped)。

---

*Benchmark artifacts and numbers were produced by running the scripts here on a 2× H100 NVL server. 数据与脚本产自 2× H100 NVL 服务器实测。*

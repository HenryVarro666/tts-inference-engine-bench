# 05 · Reproduction

> 中文速读：在 2×H100 上从零复现三步：导出主干 → eager/vLLM 基准 → TRT-LLM 容器基准。含环境、命令、坑（MPI/API 差异）。

## 0. Environment (server: 2× H100 NVL, driver 580.82.07, CUDA 13.0)

| Need | Where |
|---|---|
| source TTS checkpoint | `rednote-hilab/dots.tts-base` in HF cache (or `snapshot_download`) |
| eager + vLLM | conda env `vllm-qwen` (vllm 0.17.1, torch 2.10.0, transformers 4.57.x) |
| TRT-LLM | NGC container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23` |
| GPU pinning | `CUDA_VISIBLE_DEVICES=1` (leave GPU 0 free/other services) |

> Container access requires an NGC account + accepting the TensorRT-LLM collection EULA (pull 403/"denied" otherwise), then `docker login nvcr.io -u '$oauthtoken'` with an API key.

## 1. Export the Qwen2 backbone

`scripts/export_backbone.py` writes a standalone HF `Qwen2ForCausalLM` directory (strips `llm.` prefix from the TTS checkpoint + copies tokenizer). Verify it loads:

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("backbone_qwen2", dtype="bfloat16")
print(sum(p.numel() for p in m.parameters())/1e9, "GB")
PY
# -> 1.543 GB
```

## 2. eager + vLLM baselines (conda env `vllm-qwen`)

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/bench_eager.py 64 48   # eager, StaticCache
CUDA_VISIBLE_DEVICES=1 python scripts/bench_vllm.py 64 48    # vLLM
```

## 3. TensorRT-LLM (container)

Build engine + benchmark via the LLM API (auto-converts + runs the PyTorch backend):

```bash
docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=1 \
  -v "$PWD":/bench --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23 \
  python /bench/scripts/trtllm_bench.py
```

## 4. End-to-end latency split (eager integration)

Run the full dots-tts CLI with `--profile-inference` and read the per-stage summary from the log:

```bash
CUDA_VISIBLE_DEVICES=1 python -m dots_tts.cli \
  --model-name-or-path rednote-hilab/dots.tts-base \
  --text "..." --output out.wav --profile-inference
# -> "Inference profiling: ... stage=FM seconds=18.56 ... stage=LLM seconds=5.69 ..."
```

## Implementation gotchas (the ones we hit)

1. **vLLM ≥ 0.1x**: `LLM.generate()` no longer accepts a bare `prompt_token_ids=` kwarg — pass a list of dicts: `llm.generate([{"prompt_token_ids": [...]}], sampling_params=sp)`.
2. **TRT-LLM 1.3 removed `trtllm-build` / `convert_checkpoint`**: the LLM API auto-builds and defaults to the **PyTorch backend**. For the classical compiled-TRT-engine number, install **TRT-LLM 1.2.x** (pip, bundles TensorRT 10.14 + the old builder); PyPI offers only sdist (source build) for that line.
3. **TRT-LLM MPI**: the LLM API spawns MPI workers, so the benchmark **must** live in a real `.py` file with an `if __name__ == "__main__":` guard (a `python - <<` stdin script fails with "cannot spawn MPI from <stdin>").
4. **TRT-LLM token input**: pass `TokensPrompt(prompt_token_ids=...)` (plural `TokensPrompt`, not `TokenPrompt`).
5. **TRT-LLM PyTorch backend** rejects TRT-engine-only kwargs like `workspace`; keep kwargs minimal.
6. Container `--rm` may leave a lingering named container after an MPI abort; `docker rm -f <name>` before relaunch.

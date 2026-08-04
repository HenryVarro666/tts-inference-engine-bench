#!/usr/bin/env python3
"""vLLM benchmark: time prefill + G autoregressive tokens on a standalone Qwen2
backbone. vLLM >= 0.1x accepts token ids as a list of dicts.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prefill-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--out", default="results/vllm.json")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    P, G = args.prefill_len, args.gen_len
    llm = LLM(model=args.model_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem)
    sp = SamplingParams(max_tokens=G, temperature=0)
    prompt = list(range(10, 10 + P))
    req = [{"prompt_token_ids": prompt}]

    llm.generate(req, sampling_params=sp)  # warmup (also cudagraph/compile)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        outs = llm.generate(req, sampling_params=sp)
        times.append(time.perf_counter() - t0)
    n = len(outs[0].outputs[0].token_ids)
    best = min(times)

    res = {
        "engine": "vllm_0.17.1", "model": args.model_dir, "P": P, "G": G,
        "runs_seconds": times, "best_seconds": round(best, 4),
        "tokens_per_sec": round(n / best, 1), "per_token_ms": round(best / n * 1000, 3),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

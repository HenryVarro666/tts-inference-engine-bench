#!/usr/bin/env python3
"""TensorRT-LLM benchmark (run inside the NGC TRT-LLM container).

Uses tensorrt_llm's high-level LLM API, which auto-converts + runs. Note the
default execution backend in 1.3.0 is the PyTorch backend (the classic compiled
TRT-engine builder was removed in 1.3). Keep kwargs minimal; the PyTorch backend
rejects TRT-engine-only kwargs such as ``workspace``.

MPI caveat: this must run as a real file (not ``python - <<...``) and the work
must live under ``if __name__ == "__main__":`` so the MPI worker spawn can
re-import it.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def run(args: argparse.Namespace) -> None:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.inputs import TokensPrompt

    P, G = args.prefill_len, args.gen_len
    try:
        llm = LLM(model=args.model_dir, dtype="bfloat16", max_seq_len=args.max_seq_len)
    except TypeError:
        llm = LLM(model=args.model_dir, dtype="bfloat16")
    print("backend:", getattr(llm.args, "backend", "n/a"))

    sp = SamplingParams(max_tokens=G, temperature=0)
    prompt = list(range(10, 10 + P))
    req = TokensPrompt(prompt_token_ids=prompt)

    llm.generate(req, sampling_params=sp)  # warmup

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = llm.generate(req, sampling_params=sp)
        times.append(time.perf_counter() - t0)
    n = len(out.outputs[0].token_ids)
    best = min(times)

    res = {
        "engine": "tensorrt_llm release:1.3.0rc23 (PyTorch backend)",
        "model": args.model_dir, "P": P, "G": G,
        "runs_seconds": times, "best_seconds": round(best, 4),
        "tokens_per_sec": round(n / best, 1), "per_token_ms": round(best / n * 1000, 3),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("RESULT_JSON")
    print(json.dumps(res, indent=2))
    print("ALL_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prefill-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--out", default="results/trtllm.json")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--runs", type=int, default=3)
    run(ap.parse_args())

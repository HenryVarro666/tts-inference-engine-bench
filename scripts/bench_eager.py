#!/usr/bin/env python3
"""eager-PyTorch baseline: time prefill + G autoregressive tokens on a standalone
Qwen2 backbone using transformers StaticCache (mirrors how dots-tts drives its LM).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prefill-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--out", default="results/eager.json")
    args = ap.parse_args()

    P, G = args.prefill_len, args.gen_len
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = AutoConfig.from_pretrained(args.model_dir)
    m = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).cuda().eval()
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=P + G + G,
                        dtype=torch.bfloat16, device="cuda")

    prompt = torch.tensor([list(range(10, 10 + P))], dtype=torch.long, device="cuda")

    # prefill
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        logits = m(prompt, use_cache=True, past_key_values=cache)[0]
    torch.cuda.synchronize(); t_pre = time.perf_counter() - t0

    # decode: one token per step
    tok = cfg.bos_token_id if cfg.bos_token_id is not None else (cfg.pad_token_id or 0)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(G):
            x = torch.tensor([[tok]], dtype=torch.long, device="cuda")
            logits = m(x, use_cache=True, past_key_values=cache)[0]
            tok = int(logits[:, -1].argmax(-1).item())
    torch.cuda.synchronize(); t_dec = time.perf_counter() - t0

    res = {
        "engine": "eager_torch", "model": m.config._name_or_path, "P": P, "G": G,
        "prefill_seconds": round(t_pre, 4), "decode_seconds": round(t_dec, 4),
        "decode_per_token_ms": round(t_dec / G * 1000, 3),
        "total_generate_seconds": round(t_pre + t_dec, 4),
        "tokens_per_sec": round(G / t_dec, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

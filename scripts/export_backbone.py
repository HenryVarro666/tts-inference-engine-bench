#!/usr/bin/env python3
"""Export the autoregressive LLM backbone of a multi-stage TTS as a standalone
Hugging Face Qwen2ForCausalLM so it can be served by LLM engines (vLLM, TRT-LLM).

For dots-tts: the TTS checkpoint stores the LM under the ``llm.`` prefix. We
strip that prefix and pair it with the TTS checkpoint's ``llm_config.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from safetensors.torch import save_file
from safetensors import safe_open

LLM_PREFIX = "llm."
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="TTS checkpoint dir (HF snapshot)")
    ap.add_argument("--tgt", required=True, help="output standalone Qwen2 dir")
    ap.add_argument("--config-file", default="llm_config.json",
                    help="config file inside --src describing the LM architecture")
    ap.add_argument("--weights-file", default="model.safetensors",
                    help="safetensors file inside --src holding the LM weights")
    args = ap.parse_args()

    os.makedirs(args.tgt, exist_ok=True)

    # 1) Re-key lm.* -> standalone names
    out: dict = {}
    with safe_open(os.path.join(args.src, args.weights_file), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(LLM_PREFIX):
                out[k[len(LLM_PREFIX):]] = f.get_tensor(k)
    save_file(out, os.path.join(args.tgt, "model.safetensors"))
    print(f"exported tensors: {len(out)}")

    # 2) Write config.json
    cfg = json.load(open(os.path.join(args.src, args.config_file)))
    cfg["_name_or_path"] = "tts-qwen2-backbone"
    cfg["architectures"] = ["Qwen2ForCausalLM"]
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(args.tgt, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote config.json (vocab={cfg.get('vocab_size')})")

    # 3) Copy tokenizer files
    for fn in TOKENIZER_FILES:
        s = os.path.join(args.src, fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(args.tgt, fn))
    print("output ->", args.tgt)


if __name__ == "__main__":
    main()

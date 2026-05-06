#!/usr/bin/env python3
"""LiteRT-LM real-prompt MTP benchmark on macOS Metal GPU.

Why this exists:
  `litert-lm benchmark` decodes random tokens — drafter acceptance ~0%, so MTP
  shows as a regression. Useless for measuring MTP. This script runs real
  prompts via `litert-lm run` and times wall-clock generation, counting output
  tokens with the HF tokenizer for the matching Gemma 4 size.

Numbers are end-to-end (include ~1–2 s per-process init after cache is primed).
That's the user-perceived rate. For pure decode rate, see `bench_mtp.py`'s
transformers path which does in-process timing.

Usage:
    python bench_litertlm.py                              # E2B GPU, all prompts
    python bench_litertlm.py --model e4b                  # E4B GPU
    python bench_litertlm.py --backend cpu --model e2b    # CPU sanity check
    python bench_litertlm.py --prompts code_short
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

from transformers import AutoTokenizer


BUNDLES = {
    "e2b": ("litertlm/gemma-4-E2B-it.litertlm", "google/gemma-4-E2B-it"),
    "e4b": ("litertlm/gemma-4-E4B-it.litertlm", "google/gemma-4-E4B-it"),
}


PROMPTS: list[tuple[str, str]] = [
    (
        "code_short",
        "Write a Python function that takes a list of integers and returns the "
        "longest strictly increasing contiguous subarray. Include a docstring "
        "and one example.",
    ),
    (
        "math_chain",
        "A train leaves Mumbai at 6am going 80 km/h east. Another leaves Pune "
        "at 7am going 60 km/h west on the same track. Pune is 150 km east of "
        "Mumbai. When and where do they meet? Show your work step by step.",
    ),
    (
        "ide_completion",
        "Complete this Python function. Only provide the function body — no "
        "explanation or surrounding comments.\n\n"
        "def find_kth_largest(nums: list[int], k: int) -> int:\n"
        '    """Return the k-th largest element using a min-heap of size k.\n'
        "    Time: O(n log k), Space: O(k).\n"
        '    """\n',
    ),
]


def run_once(bundle: str, prompt: str, *, backend: str, mtp: bool) -> tuple[float, str]:
    cmd = [
        "litert-lm", "run", bundle,
        "--backend", backend,
        "--enable-speculative-decoding", "true" if mtp else "false",
        "--temperature", "0",
        "--prompt", prompt,
    ]
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if cp.returncode != 0:
        raise RuntimeError(f"litert-lm failed (exit {cp.returncode}):\n{cp.stderr[-800:]}")
    return dt, cp.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(BUNDLES), default="e2b")
    p.add_argument("--bundle", help="Override bundle path (relative or absolute)")
    p.add_argument("--tokenizer", help="Override HF tokenizer id")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--prompts", nargs="*", help="Subset by name (e.g. code_short math_chain)")
    p.add_argument("--repeats", type=int, default=1, help="Measured repeats per (prompt, mode)")
    args = p.parse_args()

    bundle = args.bundle or BUNDLES[args.model][0]
    tok_id = args.tokenizer or BUNDLES[args.model][1]
    if not Path(bundle).exists():
        raise SystemExit(f"missing bundle: {bundle}")

    selected = PROMPTS
    if args.prompts:
        selected = [(n, t) for (n, t) in PROMPTS if n in set(args.prompts)]
        if not selected:
            raise SystemExit(f"no prompts matched: {args.prompts}")

    print(f"[config] bundle={bundle}")
    print(f"[config] backend={args.backend}  tokenizer={tok_id}  repeats={args.repeats}")
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    # Two warmup invocations per mode to prime the mldrift program cache.
    print("[warmup] priming program cache (2 baseline + 2 mtp)...")
    for _ in range(2):
        run_once(bundle, "Hi.", backend=args.backend, mtp=False)
        run_once(bundle, "Hi.", backend=args.backend, mtp=True)

    rows: list[dict] = []
    for name, prompt in selected:
        for mode, mtp in (("base", False), ("mtp", True)):
            walls: list[float] = []
            n_toks: list[int] = []
            for _ in range(args.repeats):
                wall, text = run_once(bundle, prompt, backend=args.backend, mtp=mtp)
                n_tok = len(tokenizer.encode(text))
                walls.append(wall)
                n_toks.append(n_tok)
            wall = statistics.median(walls)
            n_tok = round(statistics.median(n_toks))
            tps = n_tok / wall if wall > 0 else 0.0
            rows.append({"name": name, "mode": mode, "n_tok": n_tok, "wall": wall, "tps": tps})
            label = "base " if mode == "base" else "+mtp "
            print(f"  [{name:<14}] {label} tok={n_tok:4d}  wall={wall:5.2f}s  rate={tps:6.2f} tok/s")

    print("\n=== summary ===")
    print(f"{'prompt':<18}  {'base tok/s':>10}  {'+mtp tok/s':>10}  {'speedup':>8}  "
          f"{'tok base':>9}  {'tok mtp':>9}")
    by_prompt: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_prompt.setdefault(r["name"], {})[r["mode"]] = r
    speedups: list[float] = []
    for name, d in by_prompt.items():
        if "base" in d and "mtp" in d:
            sp = d["mtp"]["tps"] / d["base"]["tps"] if d["base"]["tps"] else 0.0
            speedups.append(sp)
            print(f"{name:<18}  {d['base']['tps']:>10.2f}  {d['mtp']['tps']:>10.2f}  "
                  f"{sp:>7.2f}x  {d['base']['n_tok']:>9d}  {d['mtp']['n_tok']:>9d}")
    if speedups:
        print(f"\nmean speedup: {statistics.mean(speedups):.2f}x   "
              f"median: {statistics.median(speedups):.2f}x")
    print("\nNotes:")
    print("- tok/s is end-to-end including ~1-2s per-process init (program cache primed).")
    print("- Output lengths differ between base/mtp at temp=0 due to WebGPU sampler")
    print("  numerical nondeterminism; both outputs remain valid. Compare rates, not text.")


if __name__ == "__main__":
    main()

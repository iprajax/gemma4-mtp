#!/usr/bin/env python3
"""
Unified Gemma 4 MTP benchmark for transformers + MPS.

Single Python process. Single model load. Multiple suites & k-values.
Replaces bench_hf.py / sweep_hf.py / bench_real.py / bench_real_e2b.py.

Usage:
    python bench_mtp.py                                  # E4B, all suites, k=4
    python bench_mtp.py --model e2b                      # E2B target+drafter
    python bench_mtp.py --suites demo --k 2,4,6,8        # demo suite k-sweep
    python bench_mtp.py --suites demo,real --k 4         # both suites at k=4
    python bench_mtp.py --target google/gemma-4-E4B-it \
                        --drafter google/gemma-4-E4B-it-assistant
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers.modeling_utils as _hf_mu
_hf_mu.caching_allocator_warmup = lambda *a, **k: None  # MPS rejects giant single buffer
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ALIASES = {
    "e2b": ("google/gemma-4-E2B-it", "google/gemma-4-E2B-it-assistant"),
    "e4b": ("google/gemma-4-E4B-it", "google/gemma-4-E4B-it-assistant"),
}


# ---------- Suites: (name, prompt, output_cap_tokens) ----------

DOC_BANGALORE = """
Bangalore, the capital of Karnataka, has emerged as India's foremost technology
hub over the past three decades. Originally a garden city known for its temperate
climate and pensioner-friendly pace, it began transforming in the late 1980s when
Texas Instruments set up its first international R&D center in the country. By
the early 1990s, the Indian IT services boom — Infosys, Wipro, and later TCS — had
chosen Bangalore as their operational and talent base, drawn by the dense
concentration of engineering colleges across Karnataka and Tamil Nadu.

The 2000s saw the rise of multinational captives: IBM, Cisco, Oracle, SAP, and
Intel each established research operations employing tens of thousands of
engineers. By 2010, Bangalore had eclipsed Hyderabad and Pune in both volume of
IT export revenue and concentration of senior engineering leadership.

Starting around 2014, a third wave layered on top: domestic product startups.
Flipkart, Ola, Swiggy, Zerodha, and Razorpay grew to billion-dollar valuations,
each headquartered in Bangalore. The 2018-2021 venture funding wave deepened
this layer: by 2022, more Indian unicorns called Bangalore home than any other
Indian city by a factor of three.

The current decade has brought new pressures. Real estate prices in central
tech neighborhoods have crossed the threshold where mid-career engineers can no
longer buy first homes nearby; commute times on the Outer Ring Road regularly
exceed two hours each way during peak. Water shortages in summer 2024 forced
several office complexes to reduce on-site headcount.

Despite these strains, three trends continue to favor Bangalore as a tech hub.
First, the talent compounding effect. Second, the AI shift has rewarded cities
with deep ML talent pools. Third, capital remains sticky.
""".strip()

REVIEWS = """
Review 1 (4 stars): Battery life is genuinely excellent — I get 8-9 hours of
heavy use. Build quality feels premium. Speakers are tinny. Trackpad is best-
in-class. Display has a slight pink tint under fluorescent lights.

Review 2 (2 stars): Battery on mine drains overnight in sleep mode. Lost about
20% per night for the first month. Returned it after 90 days.

Review 3 (5 stars): Performance is unbelievable for compile workloads. Display
is gorgeous. Trackpad is the best I've ever used. Camera is mediocre.

Review 4 (3 stars): Performance and battery are great. But the fan profile is
weirdly aggressive for light tasks. Display has noticeable backlight bleed.
Software has crashed twice on me during video calls.

Review 5 (4 stars): Excellent battery, excellent display, excellent build. Fan
runs more than I'd expect. Camera mediocre. Arrow keys are cramped.
""".strip()


# Demo / headline suite — short prompts, short outputs. This is where MTP wins.
SUITE_DEMO: list[tuple[str, str, int]] = [
    (
        "code_short",
        "Write a Python function that takes a list of integers and returns the "
        "longest strictly increasing contiguous subarray. Include a docstring "
        "and one example.",
        200,
    ),
    (
        "math_chain",
        "A train leaves Mumbai at 6am going 80 km/h east. Another leaves Pune "
        "at 7am going 60 km/h west on the same track. Pune is 150 km east of "
        "Mumbai. When and where do they meet? Show your work step by step.",
        200,
    ),
    (
        "classify",
        'Classify this customer message into exactly one of: BILLING, '
        'TECHNICAL, ACCOUNT, OTHER. Reply with just the label.\n\n'
        'Message: "My internet keeps disconnecting every 10 minutes since '
        'yesterday\'s storm."',
        50,
    ),
]


# Real-world suite — production-shaped prompts. Long inputs, long outputs,
# mixed types. This is the honest number.
SUITE_REAL: list[tuple[str, str, int]] = [
    (
        "long_input_summary",
        "Read the following passage carefully, then write a 250-word summary "
        "covering: (1) the historical phases of Bangalore's tech ecosystem, "
        "(2) current pressures, and (3) the most likely future trajectory.\n\n"
        f"PASSAGE:\n{DOC_BANGALORE}\n\nSUMMARY:",
        1024,
    ),
    (
        "long_output_code",
        "Write a complete production-quality Python CLI tool that scrapes a "
        "list of URLs and stores results as JSONL. Requirements:\n"
        "- argparse with subcommands (scrape, replay, stats)\n"
        "- structured logging via logging module with file rotation\n"
        "- retry with exponential backoff on transient failures\n"
        "- concurrency via asyncio + aiohttp, configurable max workers\n"
        "- pytest tests using fixtures and mocked HTTP\n"
        "- type hints, docstrings, and a README block as a top-of-file comment\n"
        "Provide the entire file.",
        1024,
    ),
    (
        "structured_json",
        "You are an analyst. Read these laptop reviews and return a single "
        "valid JSON object with this exact schema:\n"
        "{\n"
        '  "overall_sentiment": "positive|mixed|negative",\n'
        '  "consistent_strengths": [<short phrases>],\n'
        '  "consistent_weaknesses": [<short phrases>],\n'
        '  "isolated_complaints": [<short phrases mentioned by only one reviewer>],\n'
        '  "recommended_buyer_profile": <one sentence>\n'
        "}\n"
        "Output only the JSON, no commentary.\n\n"
        f"REVIEWS:\n{REVIEWS}",
        512,
    ),
    (
        "reasoning_chain",
        "A logistics company runs three warehouses A, B, C. Inventory: A=480, "
        "B=210, C=150. Retailers needing service: R1=200, R2=280, R3=240 by "
        "end of week. Costs/unit (rupees):\n"
        "  A->R1=12  A->R2=18  A->R3=22\n"
        "  B->R1=15  B->R2=10  B->R3=14\n"
        "  C->R1=20  C->R2=16  C->R3=8\n"
        "Constraint: each warehouse's outbound shipment ≤ 60% of its inventory. "
        "Find the minimum-cost allocation. Show steps: (1) outbound caps, (2) "
        "feasibility, (3) greedy initial allocation, (4) demand check, (5) "
        "total cost, (6) one swap candidate, (7) final allocation and cost.",
        1024,
    ),
    (
        "chat_short_turn",
        "User: I'm planning a 5-day trip to Tokyo in March, mostly interested "
        "in food, design, and walkable neighborhoods. Any opinionated picks?\n"
        "Assistant:",
        512,
    ),
    (
        "ide_completion",
        "Complete this Python function. Only provide the function body — no "
        "explanation or surrounding comments.\n\n"
        "def find_kth_largest(nums: list[int], k: int) -> int:\n"
        '    """Return the k-th largest element using a min-heap of size k.\n'
        "    Time: O(n log k), Space: O(k).\n"
        '    """\n',
        200,
    ),
]


SUITES = {"demo": SUITE_DEMO, "real": SUITE_REAL}


# ---------- Core primitives ----------

@dataclass
class RunResult:
    text: str
    n_tokens: int
    secs: float
    tok_s: float

@dataclass
class ScenarioResult:
    suite: str
    name: str
    k: int
    n_in: int
    n_out: int
    base: RunResult
    spec: RunResult
    speedup: float
    hash_match: bool


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def generate(model, tokenizer, prompt: str, *, max_tokens: int,
             assistant=None, num_assistant: int = 4, device: str) -> RunResult:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    n_in = inputs["input_ids"].shape[-1]

    kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    if assistant is not None:
        kwargs["assistant_model"] = assistant
        kwargs["num_assistant_tokens"] = num_assistant

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0

    n_out = out.shape[-1] - n_in
    text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
    return RunResult(text=text, n_tokens=n_out, secs=dt, tok_s=n_out / dt if dt > 0 else 0.0)


def load_pair(target: str, drafter: str, dtype, device: str):
    print(f"[load] target  {target}")
    tokenizer = AutoTokenizer.from_pretrained(target)
    model = AutoModelForCausalLM.from_pretrained(target, dtype=dtype, device_map=device)
    model.eval()
    print(f"[load] drafter {drafter}")
    drafter_model = AutoModelForCausalLM.from_pretrained(drafter, dtype=dtype, device_map=device)
    drafter_model.eval()
    if device == "mps":
        mem_gb = torch.mps.driver_allocated_memory() / 1e9
        print(f"[load] GPU mem after load: {mem_gb:.2f} GB")
    return model, tokenizer, drafter_model


def warmup(model, tokenizer, drafter, device: str, k: int):
    print("[warmup] running short generation to seed kernels...")
    generate(model, tokenizer, "Hello, briefly say hi.", max_tokens=32,
             assistant=drafter, num_assistant=k, device=device)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def run_suite(suite_name: str, suite: list[tuple[str, str, int]],
              model, tokenizer, drafter, *, k: int, device: str) -> list[ScenarioResult]:
    print(f"\n========== suite: {suite_name}  (k={k}) ==========")
    results: list[ScenarioResult] = []
    for name, prompt, cap in suite:
        n_in = len(tokenizer.encode(prompt))
        print(f"  [{name}]  in={n_in:>4d} tok  cap={cap}")
        base = generate(model, tokenizer, prompt, max_tokens=cap, device=device)
        spec = generate(model, tokenizer, prompt, max_tokens=cap,
                        assistant=drafter, num_assistant=k, device=device)
        match = hash_text(base.text) == hash_text(spec.text)
        speedup = spec.tok_s / base.tok_s if base.tok_s else 0.0
        print(f"    baseline {base.tok_s:6.2f} tok/s  ({base.n_tokens:4d} tok, {base.secs:5.2f}s)")
        print(f"    +draft   {spec.tok_s:6.2f} tok/s  ({spec.n_tokens:4d} tok, {spec.secs:5.2f}s)"
              f"  speedup {speedup:.2f}x  match={match}")
        results.append(ScenarioResult(
            suite=suite_name, name=name, k=k, n_in=n_in, n_out=base.n_tokens,
            base=base, spec=spec, speedup=speedup, hash_match=match,
        ))
    return results


def print_table(rows: list[ScenarioResult], title: str):
    if not rows:
        return
    print(f"\n=== {title} ===")
    print(f"{'suite':<6}  {'scenario':<20}  {'k':>2}  {'in':>5}  {'out':>5}  "
          f"{'base':>7}  {'+draft':>7}  {'speedup':>8}  match")
    for r in rows:
        print(f"{r.suite:<6}  {r.name:<20}  {r.k:>2}  {r.n_in:>5}  {r.n_out:>5}  "
              f"{r.base.tok_s:>7.2f}  {r.spec.tok_s:>7.2f}  {r.speedup:>7.2f}x  "
              f"{r.hash_match}")
    if rows:
        avg = sum(r.speedup for r in rows) / len(rows)
        match_rate = sum(r.hash_match for r in rows) / len(rows)
        print(f"  mean speedup: {avg:.2f}x   |   hash-match rate: {match_rate*100:.0f}%")


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_ALIASES), default="e4b",
                   help="Shorthand for target+drafter pair (default e4b)")
    p.add_argument("--target")
    p.add_argument("--drafter")
    p.add_argument("--suites", default="demo,real",
                   help="Comma-separated: demo,real")
    p.add_argument("--k", default="4", help="Comma-separated num_assistant_tokens (default 4)")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--save", help="Optional path to write JSON summary")
    args = p.parse_args()

    target = args.target or MODEL_ALIASES[args.model][0]
    drafter = args.drafter or MODEL_ALIASES[args.model][1]
    dtype = getattr(torch, args.dtype)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    ks = [int(x) for x in args.k.split(",")]

    print(f"[config] target={target}  drafter={drafter}")
    print(f"[config] device={args.device}  dtype={args.dtype}")
    print(f"[config] suites={suites}  k_values={ks}")

    model, tokenizer, drafter_model = load_pair(target, drafter, dtype, args.device)
    warmup(model, tokenizer, drafter_model, args.device, ks[0])

    all_rows: list[ScenarioResult] = []
    for suite_name in suites:
        suite = SUITES[suite_name]
        for k in ks:
            rows = run_suite(suite_name, suite, model, tokenizer, drafter_model,
                             k=k, device=args.device)
            all_rows.extend(rows)

    # Per-suite tables
    for suite_name in suites:
        for k in ks:
            sub = [r for r in all_rows if r.suite == suite_name and r.k == k]
            print_table(sub, f"{suite_name.upper()} suite, k={k}")

    # Final consolidated headline
    print("\n========================  HEADLINE  ========================")
    for suite_name in suites:
        for k in ks:
            sub = [r for r in all_rows if r.suite == suite_name and r.k == k]
            if not sub:
                continue
            avg = sum(r.speedup for r in sub) / len(sub)
            best = max(sub, key=lambda r: r.speedup)
            worst = min(sub, key=lambda r: r.speedup)
            print(f"  {suite_name:<6}  k={k}   "
                  f"mean {avg:.2f}x   best {best.speedup:.2f}x ({best.name})   "
                  f"worst {worst.speedup:.2f}x ({worst.name})")

    if args.save:
        out = {
            "target": target, "drafter": drafter,
            "device": args.device, "dtype": args.dtype,
            "results": [
                {"suite": r.suite, "name": r.name, "k": r.k,
                 "n_in": r.n_in, "n_out": r.n_out,
                 "baseline_tok_s": r.base.tok_s, "spec_tok_s": r.spec.tok_s,
                 "speedup": r.speedup, "hash_match": r.hash_match}
                for r in all_rows
            ],
        }
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()

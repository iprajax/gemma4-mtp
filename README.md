<div align="center">

# gemma4-mtp

**Multi-Token Prediction benchmarks for Gemma 4 on Apple Silicon**

A controlled, reproducible benchmark of Google's Gemma 4 MTP drafters across the three runtimes that can run Gemma 4 locally on a Mac — at single-user batch=1, the workload shape that defines every on-device LLM running on a laptop.

[![Read on prajax.com](https://img.shields.io/badge/▶%20Read%20the%20full%20blog%20→-prajax.com-1F1B17.svg?style=for-the-badge&labelColor=8A6E45)](https://prajax.com/gemma4-mtp-for-you/)

[![License: MIT](https://img.shields.io/badge/license-MIT-CDA84E.svg?style=flat-square&labelColor=2D2823)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-7E8AA8.svg?style=flat-square&labelColor=2D2823)](https://www.python.org)
[![transformers ≥ 5.8](https://img.shields.io/badge/transformers-≥%205.8-8A6E45.svg?style=flat-square&labelColor=2D2823)](https://huggingface.co/docs/transformers)
[![litert-lm 0.11.0](https://img.shields.io/badge/litert--lm-0.11.0-5F7355.svg?style=flat-square&labelColor=2D2823)](https://github.com/google-ai-edge/LiteRT-LM)
[![Apple Silicon · macOS 26](https://img.shields.io/badge/Apple%20Silicon-macOS%2026-1F1B17.svg?style=flat-square&labelColor=2D2823)](https://www.apple.com/macbook-pro/)

<sub>📖 [Read on prajax.com](https://prajax.com/gemma4-mtp-for-you/) · 🪞 [Mirror on GitHub Pages](https://iprajax.github.io/gemma4-mtp/) · 📊 [Findings](./findings.md) · 🛠 [Reproduce](#reproduce) · ⚙️ [Methodology](#methodology)</sub>

</div>

---

## Headline result

> **LiteRT-LM Metal GPU + MTP delivers a 2.03× speedup on Gemma 4 E4B real prompts at batch=1, ~100 tok/s end-to-end on a 24 GB MacBook M4 Pro.**

This is Google's announced ~2× MTP headline — measurable, reproducible, on Apple Silicon, today. Two of the other three runtimes that can run Gemma 4 locally either capture only a fraction of that speedup or cannot engage MTP at all. The choice of runtime turns out to matter as much as the technique.

| Runtime | Gemma 4 | Baseline tok/s | + MTP tok/s | Speedup |
|---|:---:|---:|---:|---:|
| **LiteRT-LM Metal GPU** | **E4B** | 44.6 | **82.8** | **1.86× mean / 2.03× best** |
| **LiteRT-LM Metal GPU** | E2B | 84.2 | 144.3 | 1.71× mean |
| HF transformers + MPS | E4B | 12.1 | 15.1 | 1.24× |
| HF transformers + MPS | E2B | 21.9 | 22.9 | 1.04× |
| llama.cpp Metal Q8_0 | E4B | 39.0 | n/a — no MTP | — |

End-to-end real-prompt decode tok/s at `temperature=0`, single user, batch=1. Full per-prompt breakdown in [findings.md](./findings.md).

---

## Table of contents

1. [What is MTP?](#what-is-mtp)
2. [Why this benchmark exists](#why-this-benchmark-exists)
3. [Hardware](#hardware)
4. [Results](#results)
5. [Reproduce](#reproduce)
6. [Methodology](#methodology)
7. [Memory safety on 24 GB Macs](#memory-safety-on-24gb-macs)
8. [Repository structure](#repository-structure)
9. [The benchmark trap](#the-benchmark-trap)
10. [Why one runtime wins](#why-one-runtime-wins)
11. [What this benchmark is — and isn't](#what-this-benchmark-is--and-isnt)
12. [FAQ](#faq)
13. [References](#references)
14. [Citation](#citation)
15. [License](#license)

---

## What is MTP?

**Multi-Token Prediction (MTP)** is a form of speculative decoding. A small "drafter" model (~78M parameters in Gemma 4's case — about 1% of the target's compute cost) proposes *k* tokens at each step. The big "target" model verifies all *k* in a single parallel forward pass. Tokens the target agrees with are accepted; the first disagreement halts the chain. The target gets one bonus token "for free" as a side-effect of the parallel pass.

```
Standard decoding (sequential):
  prompt → [target] → t₁ → [target] → t₂ → [target] → t₃ → [target] → t₄ → [target] → t₅
  cost: 5 full target forward passes for 5 tokens

MTP (parallel verification):
  prompt → [drafter] → t₁,t₂,t₃,t₄ → [target verify] → ✓✓✓✕ + bonus t₅
  cost: ~1.04 target passes for up to 5 tokens   (drafter ≈ 1% of target)
```

At `temperature=0` the technique is provably **lossless** — output is byte-identical to running the target alone (modulo runtime-level FP nondeterminism on MPS / WebGPU samplers; see [findings.md](./findings.md) for the long version). The only costs are a small extra drafter forward pass per cycle and a slightly larger KV-cache footprint.

**Theoretical ceiling**: ~3–5× depending on *k* and acceptance rate. **Realistic on Mac at batch=1**: 1.5–2× on structured workloads (code, JSON, math), softer on freeform prose. **Measured here on E4B + LiteRT-LM**: 2.03× best, 1.86× mean.

For an animated visual explanation including the timeline diagram, the per-cycle pipeline, and the acceptance-rate scenario grid: see [the blog post](https://iprajax.github.io/gemma4-mtp/#01-what-multi-token-prediction-is) (Section 01).

---

## Why this benchmark exists

[Google announced MTP drafters for Gemma 4 on May 5, 2026](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/), citing up to **3× decode speedup** without quality degradation. The announcement names seven frameworks (LiteRT-LM, MLX, transformers, vLLM, SGLang, Ollama, AI Edge Gallery) and includes one specific caveat for Apple Silicon:

> "the 26B mixture-of-experts model presents *unique routing challenges at a batch size of 1 on Apple Silicon*… batch sizes of 4 to 8 unlock up to a ~2.2× speedup locally."

The phrasing implies muted on-edge gains at batch=1. **Batch=1 is the on-edge default** — every single-user laptop, phone, or Pi workload runs at batch=1. This benchmark answers a question the announcement leaves open: how much of the headline reproduces on an Apple Silicon Mac, at batch=1, across the three runtimes that can actually run Gemma 4 locally?

The short answer is: *the full headline reproduces, in one runtime specifically*. The longer answer involves one non-obvious benchmark trap that accounts for why most measurements will show MTP as a no-op or a regression on Mac — the bundled synthetic decode tool is the wrong tool for measuring speculative decoding.

---

## Hardware

Every benchmark in this repository was measured on a single machine.

```
Model:        MacBook Pro · Mac16,8
Chip:         Apple M4 Pro
Cores:        12 total (8 performance + 4 efficiency)
GPU:          Apple integrated, ~16-20 cores depending on bin
Memory:       24 GB unified
Storage:      460 GB SSD
OS:           macOS 26.3.1 (build 25D771280a)
Python:       3.12.12 (uv-managed)
Workload:     single user, batch = 1
```

**Note on M4 Pro vs M4 Max.** Google's published LiteRT-LM benchmark on Gemma 4 E2B GPU shows **160 tok/s** decode. That measurement was on an M4 Max, which has roughly 1.7× more GPU cores than the M4 Pro tested here. The E2B baseline of ~84 tok/s in this repo is not a regression against Google's 160 — it's the same runtime on smaller silicon. **Speedup ratios should hold across SKUs; absolute throughput will scale with GPU core count.**

---

## Results

### Headline cross-tabulation

End-to-end decode tok/s, real prompts, `temperature=0`, batch=1, M4 Pro 24 GB:

| Runtime | Weights | Model | Baseline | + MTP | Speedup |
|---|:---:|:---:|---:|---:|---:|
| **LiteRT-LM Metal GPU** | int4 | **E4B** | 44.6 | **82.8** | **1.86× mean / 2.03× best** |
| **LiteRT-LM Metal GPU** | int4 | E2B | 84.2 | 144.3 | 1.71× mean |
| LiteRT-LM CPU | int4 | E2B | ~37 | not retested | — |
| LiteRT-LM CPU | int4 | E4B | ~24 | not retested | — |
| HF transformers + MPS | bf16 | E4B | 12.1 | 15.1 | 1.24× |
| HF transformers + MPS | bf16 | E2B | 21.9 | 22.9 | 1.04× |
| llama.cpp Metal | Q8_0 | E4B (short ctx) | 39.0 | n/a — no MTP arch | — |
| llama.cpp Metal | Q8_0 | E4B (8K ctx) | 32.4 | n/a — no MTP arch | — |
| mlx-lm | 4-bit / bf16 | both | — | — | drafter loader gap |
| mlx-swift-lm | bf16 | both | — | — | uses N-gram speculative, not MTP |
| vLLM | various | both | — | — | Linux/CUDA only |

### Per-prompt detail — LiteRT-LM Metal GPU

**E4B** (real prompts, end-to-end including ~1.2 s per-process init):

| Prompt | Output tok | Baseline tok/s | + MTP tok/s | Speedup |
|---|---:|---:|---:|---:|
| code_short | 621 / 764 | 49.74 | 100.87 | **2.03×** |
| math_chain | 984 / 947 | 51.49 | 99.81 | **1.94×** |
| ide_completion | 79 / 79 | 32.68 | 47.69 | 1.46× |

**E2B**:

| Prompt | Output tok | Baseline tok/s | + MTP tok/s | Speedup |
|---|---:|---:|---:|---:|
| code_short | 783 / 904 | 90.61 | 162.33 | **1.79×** |
| math_chain | 1176 / 1091 | 90.18 | 155.53 | 1.72× |
| ide_completion | 143 / 149 | 71.81 | 115.05 | 1.60× |

### Per-prompt detail — transformers + MPS

**E4B k-sweep** (200-tok outputs, `bench_mtp.py --suites demo --k 2,4,6,8`):

| k | code_short | math_chain | mean |
|:---:|---:|---:|---:|
| 2 | 1.34× | 1.84× | 1.59× |
| **4** | **1.89×** | **2.22×** | **2.06×** |
| 6 | 1.91× | 2.21× | 2.06× |
| 8 | 1.91× | 2.20× | 2.06× |

**E4B real workloads** (1024-tok outputs, k=4):

| Scenario | Baseline | + MTP | Speedup |
|---|---:|---:|---:|
| Long input → summary (222 tok) | 11.0 | 13.0 | 1.19× |
| Long-form code (1024 tok) | 13.0 | 13.7 | 1.05× |
| Structured JSON extract (171 tok) | 13.0 | 19.8 | **1.52×** |
| Long math reasoning (1024 tok) | 11.4 | 13.8 | 1.20× |
| **Mean** | **12.1** | **15.1** | **1.24×** |

**E2B real workloads** show 1.04× mean (essentially nothing, with two scenarios <1.0×). See [findings.md](./findings.md#test-1c-mtp-on-gemma-4-e2b-smaller-target) for the full table.

### llama.cpp KV-cache sweep

`llama.cpp` cannot engage MTP for Gemma 4 (no `gemma4_assistant` architecture in mainline as of May 2026). Baseline-only KV-cache decode tok/s on E4B:

| K / V cache | Short ctx (256 / 200) | 8K ctx (8K / 200) |
|---|---:|---:|
| f16 / f16 | 41.4 | 28.2 |
| **q8_0 / q8_0** | 38.7 | **32.4** ← best at 8K |
| q8_0 / turbo3 | 34.4 | 29.4 |
| turbo3 / turbo3 | 21.5 | 30.1 |

`q8_0/q8_0` KV beats `f16/f16` at 8K context by ~15% — quantized cache reads less memory bandwidth, which dominates compute at long context. A free win that requires no MTP at all.

---

## Reproduce

### Prerequisites

- macOS 14+ on Apple Silicon (M1 or later)
- ~10 GB free disk for model bundles + venv
- Python 3.12+ (recommended via [`uv`](https://docs.astral.sh/uv/))
- (Optional) [`gh`](https://cli.github.com/) for cloning

### One-time setup

```bash
git clone https://github.com/iprajax/gemma4-mtp
cd gemma4-mtp

# Python venv for the transformers path
python3.12 -m venv .venv
source .venv/bin/activate
pip install "transformers>=5.8" "torch>=2.4" "accelerate" "tokenizers"

# LiteRT-LM CLI for the LiteRT-LM path
uv tool install litert-lm
```

The first `bench_litertlm.py` run will download the LiteRT-LM Gemma 4 bundle from Hugging Face (~3.4 GB for E4B, ~2.4 GB for E2B). The first `bench_mtp.py` run will download the bf16 transformers weights from Hugging Face (~16 GB for E4B target + ~160 MB for the drafter).

### Run the LiteRT-LM benchmark (recommended first run)

```bash
# E2B is smaller and runs in ~60 s
python bench_litertlm.py --model e2b

# E4B for the headline number, ~3 minutes
python bench_litertlm.py --model e4b
```

Expected output:

```
[config] bundle=litertlm/gemma-4-E4B-it.litertlm
[config] backend=gpu  tokenizer=google/gemma-4-E4B-it  repeats=1
[warmup] priming program cache (2 baseline + 2 mtp)...
  [code_short    ] base  tok= 621  wall=12.49s  rate= 49.74 tok/s
  [code_short    ] +mtp  tok= 764  wall= 7.57s  rate=100.87 tok/s
  [math_chain    ] base  tok= 984  wall=19.11s  rate= 51.49 tok/s
  [math_chain    ] +mtp  tok= 947  wall= 9.49s  rate= 99.81 tok/s
  [ide_completion] base  tok=  79  wall= 2.42s  rate= 32.68 tok/s
  [ide_completion] +mtp  tok=  79  wall= 1.66s  rate= 47.69 tok/s

=== summary ===
prompt              base tok/s  +mtp tok/s   speedup   tok base    tok mtp
code_short               49.74      100.87     2.03x        621        764
math_chain               51.49       99.81     1.94x        984        947
ide_completion           32.68       47.69     1.46x         79         79

mean speedup: 1.81x   median: 1.94x
```

If your `+mtp` rates are within ~5% of the `base` rates, MTP is not actually engaging. **See [The benchmark trap](#the-benchmark-trap) below** — most often, the cause is using the bundled synthetic-decode benchmark instead of a real-prompt one.

### Run the transformers benchmark

```bash
source .venv/bin/activate

# Demo k-sweep on E4B (200-tok outputs, ~10 minutes)
python bench_mtp.py --model e4b --suites demo --k 2,4,6,8

# Real workloads on E4B at k=4 (1024-tok outputs, ~15 minutes)
python bench_mtp.py --model e4b --suites real --k 4

# Save results to JSON
python bench_mtp.py --model e4b --suites demo --k 4 --save my_results.json
```

### Verify MTP is actually engaged

The single highest-leverage diagnostic: the engine config flag in the verbose log.

```bash
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  --backend=gpu \
  --enable-speculative-decoding=true \
  --temperature=0 \
  --verbose \
  --prompt="Hello" 2>&1 | grep "Speculative decoding"
```

Expected on a working setup:

```
Speculative decoding       : true
```

If it prints `false` while you passed `=true`, the engine isn't actually engaging the drafter — re-pull the bundle from Hugging Face and try again.

---

## Methodology

The numbers in this repo come from a deliberately simple measurement protocol designed to avoid the two most common ways MTP benchmarks mislead.

### What gets measured

For each `(runtime, model, mode)` tuple, the benchmark records:

- **Output tokens** counted via the matching Hugging Face tokenizer on the actual decoded string. Not the engine's internal token count, which can differ for special tokens.
- **Wall-clock time** measured externally with `time.perf_counter()` around the generation call (transformers) or around the subprocess invocation (LiteRT-LM).
- **Throughput** computed as `n_tokens / wall_clock`, end-to-end. The LiteRT-LM numbers include ~1.2 s of per-process init that's common to both baseline and MTP runs, so reported speedups are slightly conservative; decode-only speedups are ~10% higher.
- **Output equality hash** at greedy decoding. SHA-256 of the decoded string. At `temperature=0` baseline and +MTP outputs should hash-match for short outputs (≤200 tokens). MPS bf16 nondeterminism causes divergence at longer outputs — this is a runtime property, not a drafter regression. See [findings.md](./findings.md).

### What is held constant

- **Hardware**: same MacBook, same firmware, same OS, same charging state (plugged in).
- **Prompts**: three real prompts per benchmark (code generation, multi-step word problem, IDE-style completion). Prompts are checked into the scripts; they don't drift.
- **Decoding**: greedy, `temperature=0`, no top-k / top-p / repetition penalty. Maximum determinism the runtime permits.
- **Warmup**: each benchmark runs two warmup invocations per mode before any measurement, to prime the program cache (LiteRT-LM) or kernel cache (MPS).
- **Process boundaries**: each LiteRT-LM measurement is a fresh subprocess (the CLI has no Python API). Init overhead is therefore in the wall-clock; this is the realistic on-edge usage pattern.

### What deliberately is not held constant

- **Output length** between baseline and +MTP. Outputs differ slightly between modes due to runtime FP nondeterminism (sage vs WebGPU samplers behave differently in the last bit). Tokens are counted on each actual output, so the rate measure is fair.
- **Bundle dates** for LiteRT-LM. The script downloads from `litert-community/gemma-4-*-litert-lm` on first run; if the upstream re-cuts the bundle, the next clean run picks up the new metadata. This is intentional — the test is "does MTP work today, on the current bundle."

### Why synthetic benchmarks lie

The bundled `litert-lm benchmark` command decodes random tokens for its synthetic decode loop. On random sequences, drafter acceptance rate is ~0%, so MTP becomes pure verification overhead and reads as a regression. A correctly engaged MTP path scored 0.93× through this tool versus 1.79–2.03× through real prompts in the same session. **Never use synthetic-token benchmarks to measure speculative decoding.** This is what `bench_litertlm.py` exists to fix.

---

## Memory safety on 24 GB Macs

**Read this before running E4B benchmarks on a 24 GB machine.**

Gemma 4 E4B bf16 (transformers) target + drafter occupy approximately **17.4 GB of Metal-wired memory**. The default Metal wired-memory cap on 24 GB Apple Silicon is ~16–18 GB, and macOS swap is typically off — once wired pressure exceeds the cap, the kernel has nowhere to spill and **panics into a forced reboot**. This has happened during development of this repo.

The danger combinations:

- ❌ `bench_mtp.py --model e4b --suites demo,real --k 2,4,6,8` (multi-suite × multi-k on E4B in one process)
- ❌ Running E4B with Chrome / Slack / Docker open (each can hold 1–5 GB)
- ❌ Running E4B with another Python process that has bf16 weights resident

The safe combinations:

- ✅ `bench_mtp.py --model e4b --suites demo --k 2,4,6,8` — short outputs, KV stays small
- ✅ `bench_mtp.py --model e4b --suites real --k 4` — long outputs but single k
- ✅ `bench_mtp.py --model e2b ...` — E2B fits comfortably regardless of suite/k combination
- ✅ `bench_litertlm.py --model e4b` — int4 weights, ~5 GB peak, well under the cap

Pre-flight checklist for any E4B transformers run:

```bash
# 1. Quit Chrome and other multi-GB apps
osascript -e 'tell application "Google Chrome" to quit'

# 2. Optional: raise the Metal wired-memory cap to 20 GB (resets on reboot)
sudo sysctl iogpu.wired_limit_mb=20480

# 3. Verify free memory
top -l 1 -n 0 -s 0 | grep PhysMem
# Want at least ~6 GB unused before launching
```

The LiteRT-LM path (int4 bundles ~3.4 GB) does not trigger this issue.

---

## Repository structure

```
gemma4-mtp/
├── README.md                  ← this file
├── LICENSE                    ← MIT
├── CITATION.cff               ← citation metadata for the GitHub "Cite" widget
├── .gitignore                 ← excludes models/, .venv/, *.litertlm, etc.
├── index.html                 ← single-file blog post with animated MTP explainer
├── findings.md                ← long-form technical writeup
├── bench_mtp.py               ← transformers + MPS MTP benchmark
├── bench_litertlm.py          ← LiteRT-LM real-prompt MTP benchmark
└── results/
    ├── bench_mtp_demo.json    ← E4B demo k-sweep results, JSON
    ├── litertlm_e2b.txt       ← LiteRT-LM E2B benchmark log
    └── litertlm_e4b.txt       ← LiteRT-LM E4B benchmark log
```

### File-by-file

- **`index.html`** — standalone blog post, ~30 KB, single file with embedded CSS animations. Opens in any browser. Mobile-first. Includes an animated MTP explainer (sequential vs parallel timelines, per-cycle pipeline, acceptance-rate scenario grid, throughput math card), the cross-tab result table, and copy-paste reproduce commands. Live version: https://iprajax.github.io/gemma4-mtp/

- **`findings.md`** — long-form technical writeup. Detailed per-prompt results for transformers (k-sweep, E4B real workloads, E2B real workloads, IDE completion), the LiteRT-LM "first attempt looked like a no-op" narrative including the synthetic-bench trap, the llama.cpp KV-cache sweep, the full cross-tabulation, and recommendations by use-case. Roughly 4× the depth of the blog.

- **`bench_mtp.py`** — transformers + MPS MTP benchmark. Loads target + drafter once into Metal, runs paired baseline / +MTP generation across `(suite, k)` configurations, reports per-prompt rates and a SHA-256 output-equality check.
  - Suites: `demo` (3 prompts, 50–200 tok output) and `real` (6 prompts, 171–1024 tok output).
  - Models: `--model e2b` or `--model e4b`. Override individually with `--target` and `--drafter`.
  - k-sweep: `--k 2,4,6,8` (any comma-separated list).
  - Save results: `--save out.json`.
  - Memory note: monkey-patches `transformers.modeling_utils.caching_allocator_warmup` to a no-op because MPS rejects the giant single buffer it tries to allocate. Keep that patch when editing.

- **`bench_litertlm.py`** — LiteRT-LM real-prompt MTP benchmark. Subprocess wrapper around `litert-lm run` (the CLI has no Python API). For each `(prompt, mode)`, spawns `litert-lm run`, times wall-clock, captures stdout, and counts output tokens with the matching Hugging Face tokenizer.
  - Models: `--model e2b` or `--model e4b`. Override with `--bundle` (local path) and `--tokenizer` (HF id).
  - Backends: `--backend gpu` (Metal) or `--backend cpu`.
  - Prompts: subset by name with `--prompts code_short math_chain`.
  - Repeats: `--repeats N` for median over N runs per `(prompt, mode)`.
  - Each run primes the program cache with two warmup invocations per mode before measuring. Numbers are end-to-end including ~1.2 s per-process init.

- **`results/`** — output artifacts from clean runs of the benchmarks above. The JSON file is from `bench_mtp.py --save`; the `.txt` files are filtered logs from `bench_litertlm.py`.

---

## The benchmark trap

The single most common reason MTP looks broken on Apple Silicon: **the default measurement tool decodes random tokens, which makes speculative decoding look like a regression even when it's wired correctly.**

### Why synthetic-token benchmarks are wrong for speculative decoding

The bundled `litert-lm benchmark` command fills its decode loop with random token IDs. There is no real continuation for the drafter to predict — the drafter's per-token acceptance rate on random sequences approaches **0%**, so every drafted token gets rejected and the only thing MTP adds is verification overhead. A correctly engaged MTP path scores **0.93×** through this tool versus **1.79–2.03×** through real prompts on the same machine, same model, same flags. The synthetic benchmark gets the technique exactly backwards: the better the drafter wiring, the worse the synthetic number, because more rejected drafts means more wasted compute.

```bash
# DON'T trust this for MTP measurement (synthetic-token decode → 0% acceptance):
litert-lm benchmark gemma-4-E4B-it.litertlm --backend gpu --enable-speculative-decoding true

# DO use real prompts via bench_litertlm.py — externally timed,
#    output tokens counted with the matching HF tokenizer:
python bench_litertlm.py --model e4b
```

`bench_litertlm.py` is a 110-line subprocess wrapper around `litert-lm run` that times wall-clock externally and counts output tokens with the matching Hugging Face tokenizer. That's the measurement that produces the 2.03× headline above.

### Sanity check that MTP is actually engaging

If your `+mtp` rates are within ~5% of the baseline rates, MTP isn't engaging. The fastest way to confirm:

```bash
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  --backend=gpu \
  --enable-speculative-decoding=true \
  --temperature=0 \
  --verbose \
  --prompt="Hello" 2>&1 | grep "Speculative decoding"
# Expected on a working setup: Speculative decoding : true
```

If the engine line prints `false` while the CLI flag was `true`, re-pull the bundle from Hugging Face — the runtime will pick up current MTP wiring metadata on the next download.

---

## Why one runtime wins

The MTP technique is identical across all three runtimes — same drafter weights, same verification step, same `k=4` default. The runtime carries the rest. Three structural differences explain the gap:

1. **int4 weights vs bf16.** LiteRT-LM uses int4-quantized weights; transformers uses bf16. Each target step is faster on int4 because less memory bandwidth is needed per parameter read. Faster targets leave more headroom for accepted-draft savings to register as wall-clock wins.

2. **Purpose-built Metal kernel chain.** LiteRT-LM's WebGPU sampler and drafter pipeline are designed for this exact pattern — drafter forward, target verify, sample, advance KV cache. The transformers `assistant_model` machinery on MPS is a generic implementation that works for any drafter architecture and pays per-step overhead for that flexibility.

3. **No Python in the hot loop.** transformers' `assistant_model` wrapper executes Python code per generated token: tensor copies, dtype conversions, generation-config dispatch. LiteRT-LM stays in compiled C++ from prefill through end-of-stream; the only Python is the CLI argument parser at startup.

The single most useful takeaway from the experiment: **for Gemma 4 on edge, runtime choice is at least as important as model choice.** Two runtimes, identical hardware, identical model, identical technique — one delivers the headline, the other delivers a fifth of it.

---

## What this benchmark is — and isn't

### It is

✅ A reproducible, single-machine, real-prompt measurement of Gemma 4 MTP across all three runtimes that can run Gemma 4 locally on Apple Silicon today (transformers + MPS, LiteRT-LM, llama.cpp), at single-user batch=1, decode-focused.

✅ A documented benchmark trap (the bundled synthetic-token decode tool reads MTP as a regression) that accounts for why most public reports of "MTP doesn't work on Mac" are measurement artifacts.

✅ An honest comparison: same prompts, same hardware, same temperature, paired baseline / +MTP runs, externally counted output tokens.

### It is not

❌ A multi-batch / serving benchmark. Batch>1 changes the picture significantly — see [Google's blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) for batched Apple Silicon numbers.

❌ A measurement of mlx-lm. mlx-lm has no `gemma4_assistant` architecture (the Gemma 4 drafter uses a clustered-vocab head with `num_centroids=2048`, shared embedding with target, and last-layer-activation feed — none implemented in mlx-lm at the time of this writing).

❌ A measurement of mlx-swift-lm. mlx-swift-lm@alpha has Gemma 4 + speculative decoding, but the speculative decoding is N-gram based, not Google's MTP drafter. Different technique.

❌ A measurement of vLLM, SGLang, or Ollama. vLLM is Linux + CUDA/ROCm only. The other two are not in scope of this hardware test.

❌ A LiteRT-LM CPU MTP measurement. Untested under real prompts in this repo. Synthetic-bench numbers in [findings.md](./findings.md) suggest CPU MTP is also a no-op or near-no-op, but real-prompt verification is left as future work.

❌ A memory-saving benchmark. TurboQuant+ KV compression is briefly tested for runtime baseline; the actual point of TurboQuant+ is fitting larger models or longer contexts in 24 GB, which this benchmark does not measure.

❌ A statistical study. Each `(prompt, mode)` cell is one run by default. Run with `--repeats 3` for median-over-3 stability if needed; for production-grade variance estimates, run on a controlled idle machine and increase repeats further.

---

## FAQ

**Will these numbers transfer to M4 Max / M3 / M2 / M1?**
The speedup ratios should hold (the technique is hardware-agnostic), but absolute throughput will scale with GPU core count. M4 Max should be ~1.5–1.7× higher absolute tok/s on the LiteRT-LM rows. M3 / M2 / M1 will be progressively lower in absolute terms but should show the same relative shape across runtimes.

**Why is llama.cpp ruled out for MTP?**
The Gemma 4 drafter is a `Gemma4AssistantForCausalLM` architecture: clustered-vocab output head with 2048 centroids, embedding shared with the target, and a feed from the target's last-layer activations. None of this is implemented in `llama.cpp` mainline as of May 2026. Adding it would require non-trivial GGUF metadata extensions plus new Metal kernels.

**What about Ollama?**
Ollama wraps `llama.cpp`, so the same architectural gap applies. Ollama is named in Google's MTP announcement as a supported framework, but the support is presumably for a future `llama.cpp` upstream that adds `gemma4_assistant`.

**What about batch sizes greater than 1?**
Out of scope here — every benchmark in this repo runs at batch=1, which is the on-edge default. Google's blog reports batch=4–8 unlocking ~2.2× speedup on the 26B model on Apple Silicon; the 7B-class E4B numbers at batch>1 should land somewhere similar but are untested in this repo.

**Why aren't `mlx-lm` numbers included?**
mlx-lm cannot load the Gemma 4 drafter (`mlx-community/gemma-4-E4B-it-assistant-bf16`) — it raises `Received parameters not in model: language_model.model.layers.X.self_attn.k_norm…` during load. The target loads fine, but a target alone is not an MTP measurement. Implementing `gemma4_assistant` in mlx-lm is upstream work tracked at [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm).

**Why does the LiteRT-LM speedup compress on the `ide_completion` prompt (1.46×) compared to `code_short` (2.03×)?**
The `ide_completion` output is only 79 tokens. End-to-end timing includes the ~1.2 s per-process init that's common to both baseline and +MTP. With short outputs, init dominates wall-clock, so the speedup ratio compresses. Decode-only speedup on `ide_completion` is ~2.4×; the end-to-end number is what an actual user experiences, so the table reports end-to-end.

**Why do baseline and +MTP outputs differ at `temperature=0`?**
MTP at greedy is documented as lossless. In practice, runtime-level FP nondeterminism (MPS bf16 ops aren't bitwise reproducible; LiteRT-LM's WebGPU sampler falls back to a statically-linked top-K kernel that has slight numeric differences) causes accumulated rounding noise to cross argmax boundaries on borderline tokens after a few hundred tokens. Outputs remain semantically valid; they just aren't byte-identical. On CUDA with deterministic kernels, hash equality should hold.

**How do I cite this work?**
See [Citation](#citation) below — both BibTeX and a `CITATION.cff` for GitHub's auto-citation widget are provided.

**How do I run on a different MacBook (or report results)?**
Open an issue with your hardware spec, OS version, and the output of `python bench_litertlm.py --model e2b` and `python bench_litertlm.py --model e4b`. Cross-machine numbers are valuable; this repo will accumulate them as PR-able rows.

---

## References

### Primary sources

- **[Accelerating Gemma 4: faster inference with multi-token prediction drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)** — Google's announcement of the Gemma 4 MTP drafters, May 5, 2026. The headline 2–3× speedup claim and the "batch=1 on Apple Silicon" caveat that motivated this benchmark.

- **[LiteRT-LM v0.11.0 release](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.11.0)** — Google's on-device runtime, May 5, 2026. First version with Gemma 4 MTP support. Required for the headline-reproducing rows in this benchmark.

- **[`litert-community/gemma-4-E4B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm)** — the post-2026-05-05 LiteRT-LM bundle for E4B. Bundles downloaded earlier silently disable MTP.

- **[`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)** — the E2B bundle, same bundle vintage requirements.

- **[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)** and **[`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant)** — the bf16 weights for the transformers path. ~16 GB target + ~160 MB drafter.

### Background

- **[Speculative decoding original paper (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)** — the technique MTP is built on. Reading order: this paper first, then Google's MTP blog for the Gemma 4-specific architecture.

- **[`google-ai-edge/LiteRT-LM` GitHub](https://github.com/google-ai-edge/LiteRT-LM)** — runtime source. Useful when the verbose log mentions a kernel or sampler by name.

- **[Hugging Face transformers `assistant_model` API](https://huggingface.co/docs/transformers/main/en/llm_optims#speculative-decoding)** — documentation for `generate(assistant_model=..., num_assistant_tokens=...)` on the transformers side.

### Adjacent work

- **[TurboQuant+ at `tqp-v0.1.1`](https://github.com/turboquant/turboquant-plus)** — community llama.cpp fork with KV-cache compression. Not MTP, but the other major Gemma 4 optimization; included for context in [findings.md](./findings.md).

- **[Apple MLX](https://github.com/ml-explore/mlx)** and **[mlx-lm](https://github.com/ml-explore/mlx-lm)** — Apple's MLX framework. Currently lacks the `gemma4_assistant` architecture, blocking MTP for Gemma 4 in this stack.

---

## Citation

If you reference these measurements in academic work, blog posts, or other repositories, please cite this repo. A `CITATION.cff` file is included so GitHub auto-generates the "Cite this repository" widget.

```bibtex
@misc{tiwari2026gemma4mtp,
  author       = {Tiwari, Pradeep},
  title        = {gemma4-mtp: Multi-Token Prediction benchmarks for Gemma 4 on Apple Silicon},
  year         = {2026},
  month        = may,
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/iprajax/gemma4-mtp}},
  note         = {Single-machine, batch=1, real-prompt benchmarks across LiteRT-LM, transformers + MPS, and llama.cpp on a MacBook M4 Pro 24 GB.}
}
```

For shorter inline citation:

> Tiwari, P. (2026). *gemma4-mtp: Multi-Token Prediction benchmarks for Gemma 4 on Apple Silicon* [Computer software]. GitHub. https://github.com/iprajax/gemma4-mtp

---

## Contributing

Cross-machine result PRs are welcome. To submit numbers from your hardware:

1. Run `python bench_litertlm.py --model e2b` and `python bench_litertlm.py --model e4b` on a freshly-rebooted machine with no other major apps open.
2. Open an issue with the output, your hardware spec (`system_profiler SPHardwareDataType | grep -E "Model|Chip|Memory"`), and your OS version.
3. If the speedup ratios differ materially from this repo's M4 Pro numbers, attach the verbose log (`--verbose`) so the engine config flag and bundle dates can be verified.

For methodology corrections or runtime additions, please open an issue first to discuss before submitting a PR.

---

## Acknowledgments

- **Google AI Edge team** for shipping LiteRT-LM v0.11.0 with the MTP drafter wiring on the same day Google's MTP announcement landed. The tight integration is what makes the headline reproducible on Mac.
- **Hugging Face transformers maintainers** for the `assistant_model` API that made the cross-runtime comparison possible.
- **The `litert-community` Hugging Face org** for shipping the Gemma 4 LiteRT-LM bundles with MTP wiring metadata that makes the headline reproducible end-to-end on a single laptop.

---

## License

Released under the [MIT License](./LICENSE). Free for commercial and personal use.

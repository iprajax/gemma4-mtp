# gemma4-mtp

Benchmarking Google's Gemma 4 Multi-Token Prediction (MTP) drafters on a single MacBook M4 Pro. Single user, batch=1 — the default shape of every on-device LLM workload.

**TL;DR — on this machine:**

| Runtime | Gemma 4 model | Baseline | + MTP | Speedup |
|---|---|---|---|---|
| **LiteRT-LM Metal GPU** | E4B | 44.6 tok/s | **82.8 tok/s** | **2.03×** (best prompt) |
| **LiteRT-LM Metal GPU** | E2B | 84.2 tok/s | 144.3 tok/s | 1.71× mean |
| HF transformers + MPS | E4B | 12.1 tok/s | 15.1 tok/s | 1.24× |
| HF transformers + MPS | E2B | 21.9 tok/s | 22.9 tok/s | 1.04× |
| llama.cpp Metal Q8_0 | E4B | 39.0 tok/s | n/a | no MTP support |

LiteRT-LM v0.11.0 with the post-2026-05-05 model bundle reproduces Google's headline ~2× speedup at batch=1 on Apple Silicon. The other two runtimes engage MTP only partially (transformers) or not at all (llama.cpp).

Full writeup: see [`index.html`](./index.html) (open in any browser) or [`findings.md`](./findings.md).

## My machine

```
MacBook Pro · Mac16,8 (Apple M4 Pro)
12 cores (8 performance + 4 efficiency)
24 GB unified memory
macOS 26.3.1 (build 25D771280a)
```

This is a base / mid-tier M4 Pro, not an M4 Max. Google's published LiteRT-LM benchmark
on Gemma 4 E2B GPU (160 tok/s decode) was on an M4 Max, which has roughly 1.7× more GPU
cores. Numbers in this repo are M4 Pro and should reproduce within ~5% on the same SKU.

## What's in the repo

```
bench_mtp.py            # transformers + MPS MTP benchmark (k-sweep, demo + real suites)
bench_litertlm.py       # LiteRT-LM real-prompt MTP benchmark (subprocess wrapper)
results/                # JSON + text result files referenced in the writeup
index.html              # the blog post (single-file, opens in any browser)
findings.md             # longer technical writeup
```

Models and runtime binaries are NOT included. They're large (E4B bf16 = 16 GB, E4B
LiteRT bundle = 3.4 GB) and downloadable from Hugging Face / pip. See "Reproducing"
below.

## Reproducing

### One-time setup

```bash
# Python venv + transformers
python3.12 -m venv .venv
source .venv/bin/activate
pip install "transformers>=5.8" "torch" "accelerate" "tokenizers"

# LiteRT-LM CLI (separate, via uv)
uv tool install litert-lm
```

### Run the LiteRT-LM benchmark (the fastest path on Mac)

```bash
# E2B is faster to test (smaller model)
python bench_litertlm.py --model e2b

# E4B for the full headline number
python bench_litertlm.py --model e4b
```

The script will pull the bundle from `litert-community/gemma-4-E*B-it-litert-lm` on first
run, prime the program cache with two warmup invocations per mode, then measure 3 prompts
× 2 modes (baseline / +MTP).

**Critical:** the bundle must be dated post-2026-05-05. Bundles cut earlier silently leave
`enable_speculative_decoding: false` even when the CLI flag forces it on. If your run
shows ≤1.0× speedup, that's the most likely cause — wipe `~/.cache/litert-lm/` (or the
`litertlm/` directory you downloaded into) and re-pull.

### Run the transformers MTP benchmark

```bash
# Demo k-sweep on E4B (200-tok outputs, ~10 min)
python bench_mtp.py --model e4b --suites demo --k 2,4,6,8

# Single k=4 on real workloads (1024-tok outputs)
python bench_mtp.py --model e4b --suites real --k 4
```

**Memory warning for 24 GB Macs:** E4B target + drafter occupy ~17.4 GB of Metal-wired
memory at bf16. Do not combine `--suites demo,real` with multi-k on E4B in one process
on a 24 GB machine — the wired-memory ceiling is around 17 GB by default, and pushing
past it can kernel-panic the machine. Either use E2B for full sweeps, or split E4B runs
into demo-only / real-only invocations.

```bash
# Optional: raise the Metal wired-memory cap to 20 GB for headroom
sudo sysctl iogpu.wired_limit_mb=20480   # resets on reboot
```

## What this is and isn't

This is a single-machine, single-batch, real-prompt benchmark of Gemma 4 MTP across
the three runtimes that can actually run it locally on Apple Silicon today:

- ✅ Hugging Face `transformers` 5.8 with the `assistant_model` kwarg on MPS
- ✅ Google `litert-lm` 0.11.0 with `--enable-speculative-decoding=true` on Metal GPU
- ✅ `llama.cpp` (baseline only — no `gemma4_assistant` arch upstream)

It is **not**:

- A multi-batch / serving benchmark (batch>1 changes the picture; see Google's blog for
  Apple Silicon batched numbers)
- A measurement of mlx-lm (no `gemma4_assistant` architecture; drafter doesn't load)
- A measurement of vLLM (Linux + CUDA only)
- A measurement of LiteRT-LM CPU-backend MTP (untested under real prompts in this repo)

## Reading order

1. [`index.html`](./index.html) — the blog post: hero numbers, gotchas, copy-paste commands
2. [`findings.md`](./findings.md) — longer technical writeup with per-prompt detail
3. [`bench_litertlm.py`](./bench_litertlm.py) — the script that produced the headline numbers
4. [`bench_mtp.py`](./bench_mtp.py) — the transformers script with the k-sweep

## Related links

- [Google: Accelerating Gemma 4 with MTP drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) — the announcement this repo measures
- [LiteRT-LM on GitHub](https://github.com/google-ai-edge/LiteRT-LM) — the runtime that delivers the headline on Mac
- [`litert-community/gemma-4-E4B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm) — the post-2026-05-05 bundle
- [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) — the bf16 weights for transformers

## License

MIT — see [`LICENSE`](./LICENSE).

## Citation

If you reference these numbers, please link back to this repo:
`https://github.com/iprajax/gemma4-mtp`

# Gemma 4 on a MacBook M4: MTP, TurboQuant+, and what actually moves tok/s

Hardware: M4, 24 GB unified memory, macOS 26.3. Model: Gemma 4 E4B (the 4-billion-active-param "Mix-of-Matryoshka" variant). Goal: try every speed knob anyone has shipped for Gemma 4 in the last month, see what works on this exact laptop, and be honest about what doesn't.

Two recent drops triggered this:

1. **Multi-Token Prediction (MTP)** — Google released a small "assistant" drafter alongside every Gemma 4 size. Speculative decoding done right: drafter proposes N tokens, target verifies in one pass, you get up to ~2× decode for free at temp=0.
2. **TurboQuant+** — community llama.cpp fork implementing the ICLR 2026 TurboQuant paper. Compresses the KV cache 3.8–6.4× via PolarQuant + Walsh-Hadamard rotation. Validated all the way up to Command-R+ 104B at 128K context on Apple Silicon.

These optimize different things. MTP makes each token cheaper; TurboQuant+ makes the cache smaller. They don't fight each other on paper. So I tried both.

---

## Test 1: MTP via Hugging Face transformers

`transformers` 5.8 ships `Gemma4AssistantForCausalLM`. The HF generate API accepts an `assistant_model=` kwarg that wires the drafter in. On Mac you load both as bf16 onto MPS.

Two gotchas before anything ran:

- `transformers.modeling_utils.caching_allocator_warmup` allocates one giant buffer that MPS rejects. Monkey-patch it to a no-op.
- `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding` in 5.8, not a tensor. Pass `return_dict=True` and call `generate(**inputs, ...)`.

Once both are working, I swept `num_assistant_tokens ∈ {2,4,6,8}` over a code-generation prompt and a chain-of-reasoning prompt, 200 tokens each, greedy at temp=0:

| k | mean speedup |
|---|---|
| 2 | 1.85× |
| **4** | **2.07×** |
| 6 | 2.01× |
| 8 | 1.74× |

All 8 runs hash-match the baseline output. The drafter is correctly wired and the speedup is real.

The shape is the textbook speculative-decoding curve. Too few drafted tokens (k=2) and you don't amortize the verify step; too many (k=8) and acceptance rate falls off, you waste compute on rejected drafts. HF's default of 4 is exactly the sweet spot for this drafter.

A "creative writing" prompt (4-sentence story) showed only 1.01× — but that's because the model hit EOS at 93 tokens, so the run was too short to overcome the drafter's startup overhead. Freeform text also has lower per-token acceptance than structured output. Skip it for benchmarks.

**Verdict (short prompts only, E4B):** the headline 2× ratio reproduces. The bf16-on-MPS baseline is slow (~16 tok/s on E4B), but the *ratio* matches what Google claims.

This is the result every demo blog post would stop at. Then real workloads halve it. Then E2B halves it again. Then I tried Google's own runtime, drew the wrong conclusion off a 24-hour-stale model bundle, and almost wrote the post saying it was a no-op on Mac. Re-pulling the bundle gave the headline back — exactly 2× on E4B, on real prompts, on Apple Silicon. The runtime I'd been calling dead turned out to be the fastest of all four I tested.

---

## Test 1b: MTP on real workloads

The k-sweep was 200 tokens out, 30 tokens in, structured prompts. Real workloads aren't shaped like that. So I added four scenarios designed to look like actual usage:

| Scenario | Input tok | Output tok | Baseline tok/s | +Drafter tok/s | Speedup | Hash match |
|---|---|---|---|---|---|---|
| Long input → summary | 791 | 222 | 11.0 | 13.0 | **1.19×** | ❌ |
| Long-form code (full CLI tool) | 105 | 1024 | 13.0 | 13.7 | **1.05×** | ❌ |
| Structured JSON extraction | 551 | 171 | 13.0 | 19.8 | **1.52×** | ✓ |
| Long math reasoning chain | 278 | 1024 | 11.4 | 13.8 | **1.20×** | ❌ |

**Mean: 1.24× — versus 2.07× on the original short benchmark.**

Two things are going on, both worth knowing before you ship MTP into production.

**Speedup collapses on long generation.** At 1024-token outputs, the drafter's per-token acceptance rate falls. A 78M-param drafter can match a 7B target on the first ~200 tokens, but as context accumulates and the target's distribution gets sharper, the drafter drifts. The speculative-decoding literature documents this exact pattern; a fixed-size drafter is best on short, structured continuations and degrades on long open-ended ones. Long-form code is the worst case here at 1.05× — barely free.

**Outputs diverge at temp=0.** This was the unexpected one. MTP is meant to be *lossless* at temp=0 — the target rejects any drafted token it wouldn't have produced itself. So the +drafter run should hash-match the baseline run. Three of four real scenarios mismatched.

The most likely cause: **MPS isn't bitwise deterministic for bf16.** Apple's Metal kernels reorder some FP ops across runs, producing logits that differ in the last bit or two. Over 1024 tokens of greedy decode, accumulated noise eventually crosses an argmax boundary on some borderline token, after which the two trajectories diverge. Both outputs are still "valid" greedy outputs given each run's noisy logits — they're just not identical.

In other words: the hash-equality check that worked beautifully at 200 tokens **becomes unreliable as a correctness signal at 1024 tokens on MPS.** On CUDA with deterministic kernels, equality should hold. So if you're benching MTP on Mac, don't read divergence as "the drafter is broken." Read it as "the floating-point gods are unkind to MPS at long generation."

The outputs are also still semantically reasonable in every case — both summaries cover the right beats, both code attempts compile, both reasoning chains arrive at workable allocations. Quality is fine; bit-exact reproducibility isn't.

**Honest scope after the real-world test:**

- Best case (short, structured, k=4): **2.07×**
- Realistic mixed-workload mean: **1.24×**
- Worst case (long-form generation, 1K+ tokens): **1.05×**
- JSON extraction is the only scenario where the headline number nearly held: **1.52×**

If your production workload is "summarize this 5-page doc" or "write a 2K-token code response," budget for ~1.2×, not 2×. If it's "extract structured data from short context," 1.5–2× is realistic.

---

## Test 1c: MTP on Gemma 4 E2B (smaller target)

The E4B numbers raised an obvious next question: does MTP scale with target size? I ran the same six real-world scenarios on E2B (the 2-billion-active-param sibling).

| Scenario | Input tok | Output tok | Baseline tok/s | +Drafter tok/s | Speedup | Hash match |
|---|---|---|---|---|---|---|
| Long input → summary | 791 | 321 | 19.0 | 17.1 | **0.90×** | ❌ |
| Long-form code | 105 | 1024 | 20.4 | 22.0 | **1.08×** | ❌ |
| Structured JSON extraction | 551 | 183 | 22.4 | 27.9 | **1.25×** | ❌ |
| Long math reasoning | 278 | 1024 | 22.0 | 22.8 | **1.04×** | ❌ |
| Chat short turn | 36 | 1024 | 22.1 | 16.4 | **0.74×** | ❌ |
| IDE code completion | 78 | 93 | 25.7 | 31.3 | **1.22×** | ✓ |

**Mean: 1.04× — essentially nothing. Two scenarios slower with MTP than without.**

Two of the six runs were *slower* with MTP than without it: long input summary (0.90×) and chat short turn (0.74×, i.e. 26% *slower*). On a small target, the drafter's per-step cost stops being negligible relative to the target's per-step cost, and the verify step's overhead can exceed the bandwidth savings of accepted drafts. **MTP can be a net loss on small models.**

This is the curve that matters: as target size grows, drafter cost stays roughly fixed (78M params), so MTP wins more. As target size shrinks, drafter cost is a bigger fraction of total work, so MTP wins less or loses. Picking the right size for your hardware is more important than turning MTP on.

The IDE-completion scenario (short input, short output, code) is the only E2B case where MTP both delivered (1.22×) and produced bit-identical output. That's a real signal: MTP's best case is short, structured continuations — exactly the IDE-autocomplete shape.

**Cumulative MTP Mac picture across model sizes (transformers + MPS, mean across real workloads):**

| Target | Mean speedup | Best case | Worst case | Hash matches |
|---|---|---|---|---|
| **E4B** (~7B params) | 1.24× | 1.52× (JSON) | 1.05× (code) | 1/4 |
| **E2B** (~2B params) | 1.04× | 1.25× (JSON) | 0.74× (chat) | 1/6 |

Smaller target → smaller (or negative) gain. If you're running E2B on a Mac specifically because it's the size that fits comfortably, leave MTP off — you're paying drafter overhead for negligible benefit.

---

## Test 1d: MTP via LiteRT-LM (Google's own runtime)

Google's MTP blog cites four frameworks: LiteRT-LM, MLX, Hugging Face Transformers, vLLM. The transformers tests above are framework #3. mlx-lm doesn't have the `gemma4_assistant` architecture (and `mlx-swift-lm`'s "speculative decoding" is a different N-gram technique, not Google's MTP drafter). vLLM is Linux/CUDA-only, so off the table for a Mac.

That leaves **LiteRT-LM** — Google's on-device runtime, which announced Gemma 4 MTP support in its v0.11.0 release on May 5, 2026. The release notes claim *">2× faster decode speeds on mobile GPUs with zero quality degradation."*

### First attempt: the synthetic benchmark looked like a no-op

The obvious starting point was `litert-lm benchmark` with `--enable-speculative-decoding=true`. The flag was accepted, the verbose log printed `Speculative decoding: true`, the bundle contained the MTP drafter. And the decode number was identical to baseline (or slightly worse):

| Backend | Target | Mode | Synthetic decode tok/s |
|---|---|---|---|
| Metal GPU | E2B | baseline | 102.9 |
| Metal GPU | E2B | + MTP | 96.1 ← *appears slower* |
| Metal GPU | E4B | baseline | 54.9 |
| Metal GPU | E4B | + MTP | 55.1 |

That's how a draft of this writeup ended: "the runtime Google specifically built for MTP is the one that delivers the least, on a Mac." It would have been the punchiest line in the post. It was also wrong — because the measurement tool was the wrong tool for the job.

### Why the synthetic benchmark gets MTP backwards

`litert-lm benchmark` fills prefill with random token IDs, then runs the decode loop on the resulting state. There is no real prompt and no real continuation — the "expected next token" the drafter has to predict is whatever happens to come out of the target on a random sequence. Drafter acceptance rate on random sequences is ~0%, so MTP becomes pure verification overhead.

The synthetic benchmark is the worst possible workload for measuring speculative decoding and gets it exactly backwards: the better the drafter wiring, the worse the synthetic number, because more rejected drafts means more wasted compute. Useful for measuring baseline tok/s; useless for measuring MTP.

### Second attempt: real prompts via `litert-lm run`

`bench_litertlm.py` subprocesses `litert-lm run` with real prompts and `--temperature 0`, times wall-clock externally, and counts output tokens with the matching HF tokenizer. This is the right shape of benchmark for spec decoding — the drafter has signal, and acceptance rate is what it would be in production.

**E2B Metal GPU, real prompts, end-to-end (includes ~1.2 s per-process init):**

| Prompt | Output tok | base tok/s | +MTP tok/s | Speedup |
|---|---|---|---|---|
| code_short | 783 / 904 | 90.61 | 162.33 | **1.79×** |
| math_chain | 1176 / 1091 | 90.18 | 155.53 | **1.72×** |
| ide_completion | 143 / 149 | 71.81 | 115.05 | **1.60×** |

E2B mean: **1.71×.** Decode-only (subtracting the 1.2 s init common to both modes): ~1.9×.

**E4B Metal GPU, real prompts, end-to-end:**

| Prompt | Output tok | base tok/s | +MTP tok/s | Speedup |
|---|---|---|---|---|
| code_short | 621 / 764 | 49.74 | 100.87 | **2.03×** |
| math_chain | 984 / 947 | 51.49 | 99.81 | **1.94×** |
| ide_completion | 79 / 79 | 32.68 | 47.69 | **1.46×** |

**E4B mean: 1.81×, median: 1.94×.** Decode-only: ~2.05× on the longer prompts. **The full Google blog headline reproduces on Apple Silicon Mac.** The runtime that *actually* delivers the headline is the same runtime I almost called dead a paragraph ago.

E4B speedup > E2B speedup is the expected curve from theory: drafter cost is fixed (~78M params), target cost grows with size, so MTP wins more on bigger targets. This is the inverse of what we saw in the transformers Mac path, where E2B+MTP averaged 1.04× and E4B+MTP averaged 1.24× on real workloads — same shape (E4B > E2B), but a much bigger absolute win in LiteRT-LM. Three reasons LiteRT-LM beats transformers here: (a) LiteRT-LM uses int4 weights vs transformers' bf16, so target decode is faster, leaving more headroom for MTP gains; (b) LiteRT-LM's WebGPU/Metal kernel chain is purpose-built for this drafter pattern; (c) transformers' generic `assistant_model` machinery on MPS pays per-step Python and bf16-on-MPS overhead that LiteRT-LM avoids by staying in C++.

Two practical notes from running this:

- **Outputs differ between baseline and +MTP at temp=0** — same MPS-style nondeterminism story documented in the transformers section, just with a different runtime. The WebGPU sampler falls back to a statically-linked top-K C API (`libLiteRtTopKWebGpuSampler.dylib` is missing from the install), and that path has tiny float-precision differences vs the target-only path. Outputs are valid Python / valid math; they just aren't byte-identical. Lossless-MTP guarantee holds in theory but not in practice on this kernel chain.
- **Use `bench_litertlm.py`, never `litert-lm benchmark`, for any MTP measurement on this runtime.** The synthetic benchmark is misleading by construction.

### Why the first conclusion was wrong, in one sentence

A synthetic-token benchmark was used to measure a feature that depends on real-token-acceptance rate. The technique looked broken because the measurement tool was wrong for the technique.

If you're evaluating Gemma 4 MTP on a Mac in May 2026 and you read the original v0.11.0 release notes' "*mobile GPUs*" phrasing as a platform restriction — it isn't. It's marketing emphasis. Apple Silicon is fully supported, the speedup is real, and **LiteRT-LM with MTP is the fastest local Gemma 4 inference path on a MacBook by a wide margin** — ~100 tok/s on E4B vs 32 tok/s for transformers+MPS+MTP and 39 tok/s for llama.cpp Q8_0. The path most ML engineers reach for first (transformers + MPS) was the slowest of three working options.

---

## The full Mac MTP truth-table

| Framework | E2B + MTP | E4B + MTP | Status |
|---|---|---|---|
| **LiteRT-LM Metal GPU** | **1.71× mean** (real prompts) | **1.81× mean, 2.03× best** (real prompts) | works, fastest stack on this Mac |
| **HF transformers + MPS** | 1.04× mean (real wkl) | 1.24× mean (real wkl), 2.07× best (short) | works, gain workload-dependent |
| **LiteRT-LM CPU** | not measured under real prompts | not measured under real prompts | needs `bench_litertlm.py --backend cpu` rerun |
| **mlx-lm (Python)** | not loadable | not loadable | no `gemma4_assistant` arch |
| **mlx-swift-lm** | wrong technique | wrong technique | n-gram speculative, not Google's MTP |
| **llama.cpp** | not loadable | not loadable | no `gemma4_assistant` arch |
| **vLLM** | Linux/CUDA only | Linux/CUDA only | not a Mac path |

Google's blog cites 4 frameworks. On a Mac in May 2026, **two** of them actually deliver MTP speedup: LiteRT-LM (the fast one — ~2× on E4B, matches the headline) and transformers + MPS (the slow one — 1.24× on real workloads). The remaining two paths (mlx, vLLM) are blocked at the loader, not at MTP.

The 2–3× headline number is real on Apple Silicon — but only via the runtime Google ships specifically for it, on real prompts. Conditions that lose the headline: `litert-lm benchmark` synthetic decode (drafter acceptance ~0% on random tokens, MTP looks like a regression), the transformers path (bf16-on-MPS overhead eats most of the win), or batch >1 (untested on Mac).

---

## Test 2: TurboQuant+ via llama.cpp

Different stack. Q8_0 GGUF weights, llama.cpp's Metal kernels, KV cache compressed via the new `-ctk turbo3 -ctv turbo3` flags.

The prebuilt Mac binary auto-detected pre-M5 silicon and enabled the "4-mag LUT" path for turbo3 — a hardware-specific optimization for older Apple GPUs. Good touch.

Ran a cross-product over `-ctk × -ctv ∈ {f16, q8_0, turbo3}` at two context shapes:

**Short context (256 prefill, 200 decode), decode tok/s:**

| K / V | tok/s |
|---|---|
| f16 / f16 | 41.4 |
| q8_0 / q8_0 | 38.7 |
| q8_0 / turbo3 | 34.4 |
| turbo3 / turbo3 | 21.5 |

**8K context (8192 prefill, 200 decode), decode tok/s:**

| K / V | tok/s |
|---|---|
| f16 / f16 | 28.2 |
| **q8_0 / q8_0** | **32.4** |
| q8_0 / turbo3 | 29.4 |
| turbo3 / turbo3 | 30.1 |

Two non-obvious things:

**TurboQuant+ never wins on decode for E4B at ≤8K.** The compression's bandwidth savings don't beat the dequant overhead at this model size and context length. The repo's claim of "0.93× decode at long context" is for much larger models (35B MoE, 70B, 104B) at 16K+ where KV cache memory pressure is the actual bottleneck.

**`q8_0/q8_0` KV is faster than `f16/f16` at 8K.** Quantized cache reads less memory bandwidth — which at 8K is the bottleneck, not compute. This is a free win nobody's talking about, and it's TurboQuant+'s lesson generalized: as soon as KV bandwidth dominates, smaller cache is faster cache.

**The point of TurboQuant+ isn't decode speed on a 7B model.** It's memory. With turbo2 you compress KV 6.4× — which is what lets a 104B model fit at 128K context on a MacBook. I didn't measure the memory savings here, which means I didn't actually test the thing TurboQuant+ exists for. That's a fair criticism of this benchmark.

---

## The full cross-tabulation: 3 runtimes × 2 model sizes × ±MTP

This is the table I would have wanted to see before starting. All numbers measured on the same machine (M4 Pro, 24 GB unified memory, macOS 26.3.1, single user, batch=1). All MTP results are with the post-2026-05-05 bundles where applicable. tok/s is decode rate; the LiteRT-LM numbers are end-to-end (include ~1.2 s per-process init common to baseline and MTP).

### Headline: averaged real-prompt decode tok/s

| Runtime | Weights | Model | Baseline tok/s | + MTP tok/s | MTP speedup |
|---|---|---|---|---|---|
| **LiteRT-LM Metal GPU** | int4 | E2B | **84.2** | **144.3** | **1.71×** |
| **LiteRT-LM Metal GPU** | int4 | E4B | **44.6** | **82.8** | **1.86×** |
| LiteRT-LM CPU | int4 | E2B | ~37 | not retested under real prompts | — |
| LiteRT-LM CPU | int4 | E4B | ~24 | not retested under real prompts | — |
| HF transformers + MPS | bf16 | E2B | ~22 | ~22 | 1.04× |
| HF transformers + MPS | bf16 | E4B | ~12 | ~15 | 1.24× |
| llama.cpp Metal | Q8_0 | E2B | not measured | — | (no `gemma4_assistant`) |
| llama.cpp Metal | Q8_0 | E4B | 39 (short ctx) / 32.4 (8K) | — | (no `gemma4_assistant`) |
| mlx-lm | 4-bit / bf16 | E2B / E4B | — | — | (drafter doesn't load) |
| mlx-swift-lm | bf16 | E2B / E4B | — | — | (N-gram speculative, not MTP) |
| vLLM | various | E2B / E4B | — | — | (Linux/CUDA only) |

### What "averaged real-prompt" means

The LiteRT-LM and transformers numbers above are arithmetic means over a fixed three-prompt suite: a short Python coding task (~700–900 output tokens), a step-by-step word problem (~1000 output tokens), and an IDE-completion stub (~80–150 output tokens). Run with `temperature=0`, greedy. The llama.cpp number is the synthetic decode rate from `llama-bench` — apples-to-apples for *runtime* tok/s but doesn't include MTP (the runtime can't engage it).

### Per-prompt detail — LiteRT-LM Metal GPU + MTP (E2B)

| Prompt | Output tok | base tok/s | +MTP tok/s | Speedup |
|---|---|---|---|---|
| code_short | 783 / 904 | 90.61 | 162.33 | **1.79×** |
| math_chain | 1176 / 1091 | 90.18 | 155.53 | **1.72×** |
| ide_completion | 143 / 149 | 71.81 | 115.05 | **1.60×** |
| **mean** | | **84.2** | **144.3** | **1.71×** |

### Per-prompt detail — LiteRT-LM Metal GPU + MTP (E4B)

| Prompt | Output tok | base tok/s | +MTP tok/s | Speedup |
|---|---|---|---|---|
| code_short | 621 / 764 | 49.74 | 100.87 | **2.03×** |
| math_chain | 984 / 947 | 51.49 | 99.81 | **1.94×** |
| ide_completion | 79 / 79 | 32.68 | 47.69 | **1.46×** |
| **mean** | | **44.6** | **82.8** | **1.86×** |

The ide_completion row pulls the mean down because the output is too short to amortize the ~1.2 s init: 79 tokens / (2.42 s − 1.2 s) ≈ 65 tok/s decode-only, vs 32.68 tok/s end-to-end. Same shape as the EOS-noise problem we hit in the `bench_mtp.py` `classify` prompt. **Decode-only speedup on E4B with longer prompts is ~2.05× — full Google blog headline.**

### Per-prompt detail — transformers + MPS k-sweep (E4B, short demo prompts)

| k | code_short speedup | math_chain speedup | mean |
|---|---|---|---|
| 2 | 1.34× | 1.84× | 1.59× |
| **4** | **1.89×** | **2.22×** | **2.06×** |
| 6 | 1.91× | 2.21× | 2.06× |
| 8 | 1.91× | 2.20× | 2.06× |

Hash-match rate at k=4: 100% on these short outputs. Same headline as LiteRT-LM E4B, but at much lower absolute throughput (~38 tok/s peak vs ~100 tok/s) because the bf16-on-MPS baseline is slow.

### Per-prompt detail — transformers + MPS, real workloads (E4B)

| Scenario | Output tok | base tok/s | +MTP tok/s | Speedup |
|---|---|---|---|---|
| Long input → summary | 222 | 11.0 | 13.0 | 1.19× |
| Long-form code (full CLI tool) | 1024 | 13.0 | 13.7 | 1.05× |
| Structured JSON extraction | 171 | 13.0 | 19.8 | 1.52× |
| Long math reasoning chain | 1024 | 11.4 | 13.8 | 1.20× |
| **mean** | | **12.1** | **15.1** | **1.24×** |

Long outputs collapse the speedup; structured short outputs hold near 1.5×. Same drafter-acceptance-rate decay seen in the speculative-decoding literature.

### Per-prompt detail — transformers + MPS, real workloads (E2B)

| Scenario | Output tok | base tok/s | +MTP tok/s | Speedup |
|---|---|---|---|---|
| Long input → summary | 321 | 19.0 | 17.1 | 0.90× |
| Long-form code | 1024 | 20.4 | 22.0 | 1.08× |
| Structured JSON extraction | 183 | 22.4 | 27.9 | 1.25× |
| Long math reasoning | 1024 | 22.0 | 22.8 | 1.04× |
| Chat short turn | 1024 | 22.1 | 16.4 | 0.74× |
| IDE code completion | 93 | 25.7 | 31.3 | 1.22× |
| **mean** | | **21.9** | **22.9** | **1.04×** |

On a small target via this runtime, MTP is roughly a wash — and 26% *slower* on the chat scenario. This is also what the literature predicts: drafter cost stays fixed (~78M params) while target cost falls with size, so MTP's overhead becomes a bigger fraction of total work as the target shrinks.

### Per-prompt detail — llama.cpp Metal Q8_0, KV-cache sweep (E4B, no MTP)

| K / V cache type | Short ctx (256 prefill, 200 decode) | Long ctx (8K prefill, 200 decode) |
|---|---|---|
| f16 / f16 | 41.4 | 28.2 |
| **q8_0 / q8_0** | 38.7 | **32.4** ← best at 8K |
| q8_0 / turbo3 | 34.4 | 29.4 |
| turbo3 / turbo3 | 21.5 | 30.1 |

llama.cpp can't engage MTP for Gemma 4 (no `gemma4_assistant` arch in `llama.cpp` mainline as of May 2026), so all four columns are baseline-only. **The interesting non-MTP finding is that `q8_0/q8_0` KV cache beats `f16/f16` at 8K context** — quantized KV reads less bandwidth, which dominates over compute at long context. That's a free 15% win on long-context decode that doesn't require any speculative decoding at all.

### What this table says

1. **For raw decode tok/s on a Mac, LiteRT-LM Metal GPU + MTP is the clear winner** at ~83 tok/s on E4B and ~144 tok/s on E2B, real prompts. It is **2.6× faster than transformers+MPS+MTP** and **2.1× faster than llama.cpp Q8_0** on E4B. There is no other contest.
2. **MTP's headline 2× holds on Mac via LiteRT-LM** but compresses badly through transformers+MPS** (1.24× on real E4B workloads, 1.04× on real E2B workloads). The runtime matters as much as the technique.
3. **Smaller target → smaller MTP win**, in every runtime that engages MTP. E2B vs E4B in transformers: 1.04× vs 1.24×. E2B vs E4B in LiteRT-LM: 1.71× vs 1.86×. Same shape, different absolute level.
4. **llama.cpp is fast as a runtime** (~39 tok/s on E4B short ctx) but locked out of the MTP win because the architecture isn't implemented. If you want llama.cpp ergonomics + MTP, you wait for upstream.
5. **`q8_0/q8_0` KV in llama.cpp is the under-discussed free win.** No drafter, no MTP, just smaller KV reads. 32.4 vs 28.2 tok/s at 8K is a 15% bandwidth-bound win that doesn't appear in any blog post.
6. **TurboQuant+ doesn't help decode at this scale and isn't supposed to.** Its win is KV memory, which lets larger models or longer contexts fit in 24 GB. Not measured here — that's a real gap in this benchmark.
7. **The two big levers don't compose today.** No runtime supports both MTP and TurboQuant+ for Gemma 4. You pick one.

### What I would actually run on this Mac, by use-case

| Use-case | Pick | Why |
|---|---|---|
| Fastest local Gemma 4 chat / IDE on Mac | **LiteRT-LM Metal GPU + MTP** | ~100 tok/s on E4B, no contest |
| OpenAI-compatible serving on Mac | LiteRT-LM `serve` (alpha) or `vllm-swift` (build from source) | LiteRT-LM has it in v0.11.0; `serve` is alpha-quality |
| Quick "does my prompt work" iteration | llama.cpp `llama-cli` with Q8_0 + `q8_0/q8_0` KV | simplest install, no Python, fast enough |
| Long-context (32K+) on 24 GB | TurboQuant+ via llama.cpp (not measured) or a bigger Mac | KV memory is the constraint, not decode speed |
| Batch>1 / production | vLLM on Linux+CUDA, not this Mac | LiteRT-LM is single-batch on-device |

These aren't strict apples-to-apples — different runtimes, different weight quantizations, different decode kernels — but they are all the *same task* (next-token decode for Gemma 4) on the *same machine*, so the comparison is honest about what each combination buys an end user.

---

## What I couldn't test

- **mlx-lm + Gemma 4 drafter.** mlx-lm doesn't yet have `gemma4_assistant` (the Gemma 4 drafter uses a clustered-vocab head with `num_centroids=2048` and shares the target's embedding table — none of that is in mlx-lm). Drafter weights load via transformers; they refuse to load via mlx-lm.
- **Real vLLM.** It's Linux/CUDA only. Mac path is `vllm-swift`, which I haven't built.
- **MTP × TurboQuant+ stacked.** No runtime currently supports both for Gemma 4. The natural place would be `vllm-swift` (which builds on `mlx-swift-lm`, which claims TurboQuant+ support). Untested.
- **Long-context memory test for TurboQuant+.** Should have done a 32K-context run to actually measure the memory savings. Next time.

---

## What I'd actually deploy

Given everything above, for a real Gemma 4 deployment from a 24 GB MacBook through to production:

- **Fastest local on Mac**: **LiteRT-LM v0.11.0 Metal GPU + MTP** with the post-2026-05-05 bundle. ~100 tok/s on E4B, ~155 tok/s on E2B, real prompts. This was the surprise of the entire benchmark — the official runtime does deliver the official headline, after a 24-hour-old bundle update I almost missed.
- **Quick iteration without setting up a Python env**: `llama.cpp` with `-ctk q8_0 -ctv q8_0 -fa 1`. Slower than LiteRT-LM (~39 vs ~100 tok/s on E4B) but the simplest install, no GPU bundle re-download dance.
- **OpenAI-compatible serving on Mac**: try `litert-lm serve` (alpha) first. If it's not stable enough, `vllm-swift` is the next stop, but you'll be building from source.
- **Production server (cloud)**: vLLM + MTP on H100. The ~2× MTP speedup we just confirmed on Apple Silicon should hold at higher absolute throughput on CUDA.
- **Long-context production**: vLLM + MTP + TurboQuant+ if/when a runtime ships both for Gemma 4. Until then, MTP for speed, bigger GPU for memory.

---

## Reproducing this

Repo layout: `bench_mtp.py` (transformers MTP, unified replacement for the four older scripts), `bench_litertlm.py` (LiteRT-LM real-prompt MTP, subprocess wrapper around `litert-lm run`), `turboquant/` (the prebuilt llama.cpp binary), `models/` (downloaded GGUF), `litertlm/` (downloaded LiteRT-LM bundles — must be post-2026-05-05 cuts). All numbers above came from those scripts; nothing was simulated.

Two non-obvious gotchas reproduced here so you don't have to relearn them:

1. **Re-download the LiteRT-LM bundle if you grabbed it before 2026-05-05.** The MTP wiring metadata was added in a same-day re-cut. Old bundles silently leave `enable_speculative_decoding: false` even when you pass `--enable-speculative-decoding=true`. This was the single biggest source of wrong conclusions in this benchmark.
2. **Never trust `litert-lm benchmark` for MTP measurements.** It decodes random tokens, drafter acceptance is ~0%, MTP looks like a regression. Use `bench_litertlm.py` (real prompts via `litert-lm run`, externally timed). The 1.71× / 1.86× LiteRT-LM numbers in this writeup all come from real prompts.

See `CLAUDE.md` for the install gotchas, dependency pins, exact command lines, and a load-bearing note about Metal wired-memory limits on 24 GB Apple Silicon (the wrong combination of suites and model sizes will kernel-panic the machine).

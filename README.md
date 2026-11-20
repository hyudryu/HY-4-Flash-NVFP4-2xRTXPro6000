# HY-4-Flash-NVFP4 on 2× RTX PRO 6000 Blackwell

Serving the community NVFP4 build [**Avifenesh/Hy4-flash-NVFP4**](https://huggingface.co/Avifenesh/Hy4-flash-NVFP4) of Tencent's Hy4-flash (220B MoE, 512 experts / 16 active, sparse MLA + sliding-window attention, Engram n-gram memory) with vLLM. It runs on two **RTX PRO 6000 Blackwell Workstation** GPUs (SM120, 96 GB each, power-limited to **450 W**) at tensor parallel 2, with a **512K context window**. Speculative decoding uses **Tencent's own FP8 DFly drafter** from [tencent/Hy4-flash-FP8](https://huggingface.co/tencent/Hy4-flash-FP8).

> **Status: baseline.** This commit records the first working configuration and its measured speeds. The goal of this repo is to optimize from here. The FP8 sibling setup is at [hyudryu/HY-4-Flash-FP8-2xRTXPro6000](https://github.com/hyudryu/HY-4-Flash-FP8-2xRTXPro6000).

The stock `vllm/vllm-openai:hy4-flash` image refuses this model on SM120 and can't load this checkpoint. This repo has a seven-file overlay patch that fixes both, the launch recipe, and measured speeds.

## TL;DR

| | NVFP4 (this repo) | FP8 ([sibling](https://github.com/hyudryu/HY-4-Flash-FP8-2xRTXPro6000)) |
|---|---|---|
| Context window | **524,288** | 262,144 |
| KV pool | **3,440,895 tokens** | 308,058 tokens |
| Weights per GPU | **50.1 GiB** | 80.8 GiB |
| Pinned host RAM (Engram) | **~62 GiB** | ~118 GiB |
| Prefill (median, 16K–64K) | ~5,130–5,155 tok/s | ~5,200–5,370 tok/s |
| Decode, 1 user: prose / coding / counting | **75 / 103 / 132** tok/s | 62 / 87 / 106 tok/s |
| Decode, 8 users aggregate: prose / coding / counting | **287 / 372 / 439** tok/s | 211 / 274 / 347 tok/s |
| Decode at 128K context, 8 users aggregate | **290 tok/s** | — (didn't fit) |
| Average GPU power during decode | **~316 W** | ~385 W |

Decode numbers include DFly speculative decoding (5 draft tokens) and count reasoning tokens. Sampling is greedy (temperature 0).

## Hardware and software

| Component | Version |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120, 97,887 MiB each), PCIe, no NVLink |
| GPU power limit | **450 W each** (default and max is 600 W) |
| CPU / RAM | AMD Ryzen Threadripper PRO 9965WX (24C/48T), 256 GB RAM (`free` reports 215 GiB usable) |
| Driver / CUDA | 595.99.02 / CUDA 13.2 driver, CUDA 13.0 in the image |
| Image | `vllm/vllm-openai:hy4-flash` = `sha256:6b645bfbf4b89602d38f6e93c1c418516ec0bfbf54b64cd661b8075bfec01da1` |
| vLLM / FlashInfer / PyTorch | 0.1.dev25463+g77766db1f / 0.7.4 / 2.14.0+cu130 |
| Target model | [`Avifenesh/Hy4-flash-NVFP4`](https://huggingface.co/Avifenesh/Hy4-flash-NVFP4) @ `f9eac06cf022cfd3dc2e8ed423922466b2adcd87` |
| Draft model | DFly drafter from [`tencent/Hy4-flash-FP8`](https://huggingface.co/tencent/Hy4-flash-FP8) @ `bd0e136e89a9ecd5ea17668cd89cf54fa127e795` (only `model.dfly_layers.*` is read, ~0.9 GiB) |
| Base model | [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) |
| OS | Linux 7.0.0 (Ubuntu) |

## The checkpoint

`Avifenesh/Hy4-flash-NVFP4` is a data-free, weight-only quantization of the BF16 `tencent/Hy4-flash`, made by a third party, not Tencent. Its own card reports 0.66% relative weight error on the experts.

| Weight class | Format | Size | Lives in |
|---|---|---|---|
| Routed experts (41 MoE layers × 512) | NVFP4 weight-only (E2M1 + E4M3 per-16 scale + FP32 per-expert scale) | 77.8 GiB | VRAM (TP2) |
| Engram n-gram tables (4) | FP8 E4M3, one FP32 scale per row | 61.5 GiB | **Pinned host RAM** |
| `q_b_proj` / `o_proj` / `linear_gate` | FP8, 128×128 blocks | | VRAM |
| Everything else (dense layer 0, shared experts, router, indexer, norms, embeddings, LM head) | BF16, unchanged | 9.2 GiB total for this row and the one above | VRAM |
| Vision tower | BF16 | 0.9 GiB | VRAM |
| Built-in DFly drafter | NVFP4 MLPs + FP8 attention | 0.7 GiB | **not used**; replaced by the vendor FP8 drafter |

Activations stay BF16. The experts run through vLLM's Marlin W4A16 kernel (`--moe-backend marlin`), so there is no activation quantization.

## What's where: VRAM, RAM, and the context pool

**Per GPU (VRAM, 95.6 GiB usable, capped at 90.2 GiB by `--gpu-memory-utilization 0.95`):**

| Item | Size per GPU |
|---|---|
| Model weights (NVFP4 experts ~38.9 GiB + FP8/BF16 dense, attention, vision, LM head), plus the FP8 DFly drafter | **50.1 GiB** |
| KV cache pool | **36.5 GiB → 3,440,895 tokens** |
| CUDA graphs | ~0.37 GiB |
| Encoder (image) cache | 0.05 GiB, taken from the KV pool |
| Total in use (`nvidia-smi`) | ~89–91 GiB |

**Host RAM:** the four Engram tables (~165M rows × 96 each) sit in **pinned host memory**. The GPUs read them over PCIe through UVA, which is vLLM's default for this model. They're stored as FP8 with FP32 row scales, about **62 GiB** (`free` shows ~68 GB shared), roughly half of what the FP8 sibling needs (BF16 tables, ~118 GiB).

**Context pool:** the KV cache uses the packed `fp8_ds_mla` layout, the only layout the SM120 sparse-MLA kernel reads. 30 of the 42 layers are sliding-window (512 tokens), so only the 12 sparse-attention layers grow with the sequence. With `--max-model-len 524288` the pool holds **6.56 full-length requests** at once, or any mix that stays under ~3.44M tokens, for example 8 × 128K with room to spare. A 448,719-token prompt with codes planted at the start and middle was recalled exactly.

## Recipe

### 1. Download both checkpoints

```bash
huggingface-cli download Avifenesh/Hy4-flash-NVFP4 --revision f9eac06cf022cfd3dc2e8ed423922466b2adcd87
# DFly drafter: the whole drafter lives in one 0.91 GiB shard of the 268 GB FP8 checkpoint.
# (This partial download is untested here; the benchmarks used a full local copy.)
huggingface-cli download tencent/Hy4-flash-FP8 --revision bd0e136e89a9ecd5ea17668cd89cf54fa127e795 \
  --include "*.json" "*.jinja" "model-00148-of-00148.safetensors"
docker pull vllm/vllm-openai:hy4-flash
```

### 2. Run it

```bash
git clone https://github.com/hyudryu/HY-4-Flash-NVFP4-2xRTXPro6000
cd HY-4-Flash-NVFP4-2xRTXPro6000
./recipe/run.sh            # extra vLLM flags can be appended
docker logs -f hy4-flash-nvfp4
```

[`recipe/run.sh`](recipe/run.sh) bind-mounts the seven patch files read-only over the image and runs:

```bash
vllm serve Avifenesh/Hy4-flash-NVFP4 --revision f9eac06cf022cfd3dc2e8ed423922466b2adcd87 \
  --served-model-name hy4-flash-nvfp4 --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.95 --moe-backend marlin \
  --max-model-len 524288 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":5,"model":"tencent/Hy4-flash-FP8","revision":"bd0e136e89a9ecd5ea17668cd89cf54fa127e795","draft_sample_method":"greedy"}' \
  --enable-auto-tool-choice --reasoning-parser hy_v4 --tool-call-parser hy_v4 \
  --enable-prompt-tokens-details
# env: HF_HUB_OFFLINE=1 VLLM_USE_RUST_FRONTEND=1
```

Startup takes about 2 minutes of weight loading on a cold page cache (~40 s when warm), then about 5 minutes of warmup and CUDA-graph capture. Test it:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash-nvfp4","messages":[{"role":"user","content":"What is 17*23?"}]}'
```

**Do not pass:**
- `--attention-backend` (FlashMLA has no SM120 kernels).
- `--kv-cache-dtype fp8`. Leave it at `auto`; it becomes `fp8_ds_mla` automatically.
- A speculative config **without** the `model`/`revision` of the FP8 checkpoint. Without them, vLLM tries the checkpoint's own NVFP4 drafter, which this image can't load.

`draft_sample_method` is set to `greedy` explicitly because the NVFP4 `config.json` carries a `speculative_config` hint with `probabilistic`, and vLLM's DFly accepts only greedy drafts.

### Optional: running under a manager (SparkDeck, etc.)

Use the same flags, with the seven files as read-only mounts. Each source is `/path/to/repo/patch/vllm/models/<file>`; each target is `/usr/local/lib/python3.12/dist-packages/vllm/models/<file>`, for `<file>` in:

```
hy_v4_flash/nvidia/sm120.py      hy_v4_flash/nvidia/attention.py   hy_v4/nvidia/attention.py
hy_v4_flash/nvidia/model.py      hy_v4_flash/nvidia/engram.py      hy_v4_flash/nvidia/nvfp4_fold.py
hy_v4/dfly_config.py
```

In SparkDeck, cloning an FP8 deployment carries over its DFly launch controls. Make sure the speculative config is the one above.

## The patch

Files are in [`patch/`](patch/), and diffs against the image are in [`patch/diffs/`](patch/diffs/).

**SM120 attention.** These three files are identical to the [FP8 repo's patch](https://github.com/hyudryu/HY-4-Flash-FP8-2xRTXPro6000#the-sm120-patch), and follow the same approach as upstream draft [vllm#68387](https://github.com/vllm-project/vllm/pull/68387) for [vllm#68380](https://github.com/vllm-project/vllm/issues/68380):
1. **`hy_v4_flash/nvidia/sm120.py` (new):**
   - Hy4 runs on FlashInfer's `FLASHINFER_MLA_SPARSE_SM120` kernel, with the per-head learnable attention sink passed through.
   - The `fp8_ds_mla` scales are read as `arbitrary_fp32`.
2. **`hy_v4_flash/nvidia/attention.py`:**
   - Accepts compute capability 12.x.
   - Sets the sliding window after construction, so the 30 SWA layers can use `fp8_ds_mla`.
3. **`hy_v4/nvidia/attention.py`:** gives the DFly drafter the same sink-capable SM120 backend.

**NVFP4 checkpoint loading:**

4. **`hy_v4_flash/nvidia/model.py`:**
   - Accepts the checkpoint's ModelOpt `MIXED_PRECISION` config (`modelopt_mixed`), with W4A16_NVFP4 and FP8_PB_WO layers only.
   - Loads stacked `[512, …]` expert tensors, including the 1-D per-expert `weight_scale_2`.
   - Renames the FP8 block scale `weight_scale_inv` to ModelOpt's `weight_scale`.
   - Fills the unused W4A16 input scales with 1.0.
   - Loads the pre-quantized FP8 Engram tables.
5. **`hy_v4_flash/nvidia/engram.py`:** keeps the Engram row scales in **FP32**, never rounded to BF16.
6. **`hy_v4_flash/nvidia/nvfp4_fold.py` (new):** vLLM's NVFP4 MoE path keeps one global scale per expert for gate and up together. The checkpoint stores them separately, and they differ by 2× in 182 of 20,992 experts.
   - This file folds the difference into the per-16 block scales and E2M1 codes. That's exact for all but a few experts in layers 1–2.
   - For those few, E4M3 limits force rounding some block scales. Worst relative error is **3.5e-6**, about 1/1900 of NVFP4's own quantization error, and the loader logs a warning.

**External FP8 drafter:**

7. **`hy_v4/dfly_config.py`:**
   - Allows a `modelopt_mixed` target to take its DFly drafter from a different checkpoint, here the vendor FP8 one.
   - The drafter's config is derived from that checkpoint's own config, so it keeps the FP8 checkpoint's blockwise `quantization_config`, and it loads that checkpoint's weights.
   - Everything else behaves as before. The override is a module-level function plus `partial`, so the engine config still pickles across worker processes.

### Accuracy notes

- **Experts are third-party NVFP4** (weight-only, data-free). The Engram tables are FP8 per row.
- On SM120 the KV cache is FP8 (`fp8_ds_mla`), and the attention math is FP8 too. See the [FP8 repo](https://github.com/hyudryu/HY-4-Flash-FP8-2xRTXPro6000#accuracy-notes).
- **The drafter is Tencent's FP8 one**, not the checkpoint's NVFP4 one. Speculative decoding is verified by the target, so the drafter affects speed only, never output.
- Smoke tests:
  - 17×23 = 391; Canberra; a correct palindrome function; 14:35 + 2 h 50 m = 17:25.
  - No repetition loops.
  - Exact recall at 4,850 tokens and at **448,719 tokens** (two codes, start and middle).
- **No full-benchmark evaluation (GSM8K, GPQA, …) has been run yet.**

## Benchmarks

Method: [`bench/bench.py`](bench/bench.py), the same as the FP8 repo, plus a per-run `--seed-offset` so the 3.4M-token prefix cache can never serve a repeated prompt.
- Sampling is greedy, and token counts come from the server's usage stats.
- Decode tok/s is measured only while every request in the batch is decoding, so prefills are excluded. "Per user" is the median request's tokens after its first token divided by time after its first token.
- Raw results, the run log, and a 1 Hz GPU power log are in [`bench/`](bench/).

### Prompt processing (prefill), single request, warm

| Prompt tokens | Time to first token (s) | Prefill tok/s |
|---|---|---|
| 1,029 | 0.36 | 2,860 |
| 4,069 | 0.91 | 4,521 |
| 8,085 | 1.69 | 4,845 |
| 16,361 | 3.18 | 5,133 |
| 32,795 | 6.36 | 5,155 |
| 65,255 | 12.72 | 5,138 |
| 130,772 | 26.51 | 4,926 |

Median of **5 sweeps** after a JIT warm-up. The spread is one-sided: most runs land within ~1% of the best, with an occasional slow run (up to −20% at 8K–32K, and once 3× slower at 1K). The median is the baseline to optimize against; per-run data is in `results-prefill-warm{1..5}.json`.

### Decode at concurrency 1 / 2 / 4 / 8 (short prompts, 1,024 output tokens)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/decode-aggregate-dark.svg">
  <img alt="Aggregate decode throughput vs concurrency: prose 75, 123, 194, 287 tok/s; coding 103, 157, 243, 372; counting 132, 224, 313, 439 at C1, C2, C4, C8" src="docs/charts/decode-aggregate-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/decode-per-user-dark.svg">
  <img alt="Per-user decode speed vs concurrency: prose 75, 61, 49, 36 tok/s; coding 103, 80, 63, 48; counting 132, 113, 82, 56 at C1, C2, C4, C8" src="docs/charts/decode-per-user-light.svg">
</picture>

| Workload | C | Aggregate tok/s | Per-user tok/s | TTFT p50 (s) |
|---|---|---|---|---|
| Prose (short story) | 1 | 75.3 | 75.3 | 0.17 |
| | 2 | 123.1 | 61.4 | 0.25 |
| | 4 | 193.5 | 48.8 | 0.32 |
| | 8 | 286.7 | 36.0 | 0.73 |
| Coding (Python + tests) | 1 | 103.1 | 103.1 | 0.17 |
| | 2 | 157.2 | 80.1 | 0.25 |
| | 4 | 243.0 | 62.5 | 0.58 |
| | 8 | 371.6 | 47.9 | 0.75 |
| Counting (1…400 in words) | 1 | 132.4 | 132.4 | 0.17 |
| | 2 | 224.3 | 112.6 | 0.18 |
| | 4 | 312.5 | 81.7 | 0.34 |
| | 8 | 439.4 | 55.5 | 0.48 |

Medians of 2 rounds per cell. Across the whole benchmark the FP8 drafter's acceptance was **25.9% per draft token**, about **2.29 tokens per decode step** with 5 drafted. It's higher on code; a code-only smoke test reached 44%.

### Decode at context 8K → 128K (prose task after a unique N-token document, 512 output tokens)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/decode-context-dark.svg">
  <img alt="Aggregate decode throughput at long context: all five context lengths from 8K to 128K track each other, about 66-85 tok/s at C1 and 288-307 tok/s at C8" src="docs/charts/decode-context-light.svg">
</picture>

| Context | C | Aggregate decode tok/s | Per-user tok/s | TTFT p50 (s) |
|---|---|---|---|---|
| 8K | 1 | 79.3 | 79.3 | 1.7 |
| 8K | 2 | 125.6 | 62.5 | 3.3 |
| 8K | 4 | 197.8 | 40.8 | 4.2 |
| 8K | 8 | 302.9 | 30.0 | 8.8 |
| 16K | 1 | 82.1 | 82.1 | 3.2 |
| 16K | 2 | 123.6 | 53.6 | 4.7 |
| 16K | 4 | 193.0 | 37.9 | 9.5 |
| 16K | 8 | 307.3 | 21.5 | 14.2 |
| 32K | 1 | 84.5 | 84.5 | 6.3 |
| 32K | 2 | 129.8 | 51.3 | 9.8 |
| 32K | 4 | 192.0 | 26.4 | 15.8 |
| 32K | 8 | 288.0 | 14.8 | 28.5 |
| 64K | 1 | 65.5 | 65.5 | 12.7 |
| 64K | 2 | 119.2 | 42.4 | 19.5 |
| 64K | 4 | 207.3 | 18.1 | 32.1 |
| 64K | 8 | 290.8 | 9.2 | 58.0 |
| 128K | 1 | 73.9 | 73.9 | 26.7 |
| 128K | 2 | 121.0 | 38.4 | 39.3 |
| 128K | 4 | 176.5 | 10.9 | 66.3 |
| 128K | 8 | 290.3 | 4.8 | 123.8 |

Thanks to sparse attention, aggregate decode speed barely depends on context: about 290–307 tok/s at C8 from 8K to 128K. Per-user tok/s at higher C is much lower than aggregate ÷ C, because each request's decode overlaps the other requests' prefills. TTFT at high C is dominated by those queued prefills: 8 × 128K is about 1M prompt tokens.

### Power (450 W limit per GPU, 1 Hz `nvidia-smi` samples)

| Phase | GPU 0 avg / max | GPU 1 avg / max | Peak temp (GPU 0 / 1) |
|---|---|---|---|
| Prefill sweeps (5×, warm) | 412 / 442 W | 419 / 448 W | 68 / 82 °C |
| Decode, C1–C8 | 312 / 432 W | 320 / 437 W | 64 / 77 °C |
| Long-context decode | 405 / 443 W | 407 / 487 W* | 69 / 82 °C |

\* A single 1 s sample above the 450 W cap. Idle is about 20–30 W per GPU with the model loaded. GPU 1 is the top card and runs about 13 °C hotter. Median SM clocks under load were 2.69–2.84 GHz.

### Reproduce

```bash
cd bench
POOL=3440895 ./run_all.sh     # warm-up + 2 prefill sweeps + decode matrix + context sweep, with a GPU power log
python3 make_charts.py        # regenerates docs/charts/*.svg from the JSON results
```

## Optimization targets

The baseline uses defaults everywhere. Candidates, none tried yet:
- **MoE kernel:** Marlin W4A16 runs with default tuning, and the startup log says SM120 lacks native FP4, which is wrong. Try FlashInfer/CUTLASS NVFP4 weight-only paths and tuned Marlin configs. No activation quantization.
- **Sparse-MLA autotune:** there's no FlashInfer SM120 sparse-MLA autotune cache, so the kernels run with default tactics.
- **CUDA graphs:** the `FULL_AND_PIECEWISE` mode falls back. Cover more batch shapes, including speculative verification.
- **Speculative decoding:** tune `num_speculative_tokens` for 25.9% acceptance. Fewer draft tokens may be faster on prose.
- **Prefill:** tune `--max-num-batched-tokens`, and find the cause of the occasional slow prefill runs.
- **Engram over PCIe:** 62 GiB of pinned tables are read through UVA every step. With the VRAM freed up, try moving some or all of them onto the GPUs.
- **Power:** run the same sweep at 500–600 W to see how much of prefill and long-context decode is power-bound.

## Known limitations

- **Upstream support is not merged yet**; this is a local overlay on the `hy4-flash` image. Track [vllm#68380](https://github.com/vllm-project/vllm/issues/68380) and [vllm#68387](https://github.com/vllm-project/vllm/pull/68387).
- The checkpoint is a community quantization whose card says it has had no whole-model evaluation.
- The NVFP4 target and FP8 drafter come from different releases built from the same base model. Acceptance is lower than you'd expect from a matched pair, but output is unaffected.
- **Startup warnings, all benign:**
  - The Marlin "no native FP4" message.
  - No sparse-MLA autotune cache.
  - The CUDA-graph mode fallback.
  - `torch.compile` isn't supported for this model.
  - The DFly drafter ignores image embeddings.

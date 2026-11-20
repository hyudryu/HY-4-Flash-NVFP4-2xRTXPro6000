# HY-4-Flash-NVFP4 on 2× RTX PRO 6000 Blackwell

Serving the community NVFP4 build [**Avifenesh/Hy4-flash-NVFP4**](https://huggingface.co/Avifenesh/Hy4-flash-NVFP4) of Tencent's Hy4-flash (220B MoE, 512 experts / 16 active, sparse MLA + sliding-window attention, Engram n-gram memory) with vLLM. It runs on two **RTX PRO 6000 Blackwell Workstation** GPUs (SM120, 96 GB each, power-limited to **450 W**) at tensor parallel 2, with a **512K context window**. Speculative decoding uses **Tencent's own FP8 DFly drafter** from [tencent/Hy4-flash-FP8](https://huggingface.co/tencent/Hy4-flash-FP8).

> **Status: V1.** The first optimization pass over the [baseline](#v1-vs-baseline): decode is 7–46% faster and prefill about 5% faster, with the same weights and no new quantization. The FP8 sibling setup is at [hyudryu/HY-4-Flash-FP8-2xRTXPro6000](https://github.com/hyudryu/HY-4-Flash-FP8-2xRTXPro6000).

The stock `vllm/vllm-openai:hy4-flash` image refuses this model on SM120 and can't load this checkpoint. This repo has an eight-file overlay patch that fixes both, the launch recipe, and measured speeds.

## TL;DR

| | Baseline | **V1** | FP8 ([sibling](https://github.com/hyudryu/HY-4-Flash-FP8-2xRTXPro6000)) |
|---|---|---|---|
| Decode, 1 user: prose / coding / counting | 75 / 103 / 132 | **92 / 120 / 142** tok/s | 62 / 87 / 106 |
| Decode, 8 users aggregate: prose / coding / counting | 287 / 372 / 439 | **416 / 519 / 610** tok/s | 211 / 274 / 347 |
| Decode at 128K context, 8 users aggregate | 290 | **396 tok/s** | — (didn't fit) |
| Prefill (median, 16K–64K) | ~5,130–5,155 | **~5,389–5,406 tok/s** | ~5,200–5,370 |
| Decode step latency, 1 user | 26.5 ms | **20.3 ms** | — |
| Context window / KV pool | same | 524,288 / 3.45M tokens | 262,144 / 308K tokens |
| Weights per GPU / pinned host RAM | same | 50.1 GiB / ~62 GiB | 80.8 GiB / ~118 GiB |

Decode numbers include DFly speculative decoding and count reasoning tokens. Sampling is greedy (temperature 0).

## V1 vs baseline

V1 is the baseline plus four changes. None changes the weights or adds quantization.

| # | Change | What it does | Effect |
|---|---|---|---|
| 1 | **Fused RMSNorm** (`--ir-op-priority`, one flag) | vLLM picked its unfused "native" RMSNorm, expecting `torch.compile` to fuse it, but this model doesn't support `torch.compile`. So each of the ~290 norms per step ran as 8 tiny kernels. The flag selects vLLM's own fused C kernel. | ~2,000 fewer kernel launches per step; decode step 26.5 → 24.0 ms |
| 2 | **Sliding-window index cache** (`swa.py`) | All 30 sliding-window layers recomputed the same window indices every step (~30 tiny kernels each). They depend only on the batch, so they're now computed once per forward pass and copied. The result is bit-identical. | 24.0 → 23.0 ms |
| 3 | **3 draft tokens instead of 5** | Every drafted token is verified by the full model, and that cost grows with batch size. Draft positions 4–5 were accepted only 8–12% of the time. | 23.0 → 20.3 ms at 1 user; ~50 → ~36 ms at 8 users |
| 4 | **MoE kernel A/B** | b12x (SM120 W4A16) was tested against Marlin and was equivalent within noise, so V1 stays on Marlin. | none |

**How it was found:** a `torch.profiler` trace of one decode step (per-kernel GPU time on both TP ranks) showed about **10,700 kernel launches per step**. Tiny unfused elementwise ops were the largest slice at 32% of step time. Dense BF16 GEMMs took 21%, the Marlin MoE GEMM 13%, sparse-MLA attention 11%, DeepGEMM 10%, the LM head 4.5%, and TP all-reduce 4.3% with no wait gaps.

**Speculative decoding:** per-draft-token acceptance rose from 25.9% to **38.6%** over the full benchmark, because the low-yield positions are no longer drafted. Tokens per step went from 2.29 to 2.16, while each step got 23–28% cheaper. In the baseline, acceptance by position was 58% / 33% / 19% / 12% / 8%. A sweep of 2, 3, 4, 5 and 8 draft tokens put 3 on top at both 1 and 8 users.

**Output:** changes 2–4 don't alter the model's math. The fused RMSNorm computes the same function but rounds once instead of twice, so text can diverge from the baseline after a few dozen tokens. Baseline-vs-baseline runs diverge the same way. The smoke tests still pass.

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

[`recipe/run.sh`](recipe/run.sh) bind-mounts the eight patch files read-only over the image and runs:

```bash
vllm serve Avifenesh/Hy4-flash-NVFP4 --revision f9eac06cf022cfd3dc2e8ed423922466b2adcd87 \
  --served-model-name hy4-flash-nvfp4 --tensor-parallel-size 2 --dtype bfloat16 \
  --gpu-memory-utilization 0.95 --moe-backend marlin \
  --max-model-len 524288 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":3,"model":"tencent/Hy4-flash-FP8","revision":"bd0e136e89a9ecd5ea17668cd89cf54fa127e795","draft_sample_method":"greedy"}' \
  --enable-auto-tool-choice --reasoning-parser hy_v4 --tool-call-parser hy_v4 \
  --ir-op-priority '{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}' \
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

Use the same flags, with the eight files as read-only mounts. Each source is `/path/to/repo/patch/vllm/models/<file>`; each target is `/usr/local/lib/python3.12/dist-packages/vllm/models/<file>`, for `<file>` in:

```
hy_v4_flash/nvidia/sm120.py      hy_v4_flash/nvidia/attention.py   hy_v4/nvidia/attention.py
hy_v4_flash/nvidia/model.py      hy_v4_flash/nvidia/engram.py      hy_v4_flash/nvidia/nvfp4_fold.py
hy_v4/dfly_config.py             hy_v4_flash/nvidia/swa.py
```

In SparkDeck, cloning an FP8 deployment carries over its DFly launch controls. Make sure the speculative config is the one above, and set its draft-token control to 3.

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

**V1 performance:**

8. **`hy_v4_flash/nvidia/swa.py`:** caches the sliding-window indices once per forward pass.
   - It's keyed on the forward-context object and the batch's `query_start_loc` / `seq_lens` tensors, so a different pass or batch never matches.
   - The other 29 SWA layers copy the cached result.
   - A one-time log line, `Flash SWA window-index cache: hit=True`, confirms it's active.

The fused RMSNorm is a launch flag (`--ir-op-priority`), not a patch.

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
- Raw results, run logs, and 1 Hz GPU power logs are in [`bench/results/v1/`](bench/results/v1/) and [`bench/results/baseline/`](bench/results/baseline/).

### Prompt processing (prefill), single request, warm

| Prompt tokens | Baseline tok/s | V1 tok/s | Change | Time to first token (s) |
|---|---|---|---|---|
| 1,029 | 2,860 | 2,859 | -0% | 0.36 |
| 4,069 | 4,521 | 4,660 | +3% | 0.87 |
| 8,085 | 4,845 | 5,095 | +5% | 1.61 |
| 16,361 | 5,133 | 5,406 | +5% | 3.02 |
| 32,795 | 5,155 | 5,389 | +5% | 6.09 |
| 65,255 | 5,138 | 5,405 | +5% | 12.12 |
| 130,772 | 4,926 | 5,168 | +5% | 25.27 |

Median of **5 sweeps** after a JIT warm-up, for both V1 and the baseline; per-run data is in `bench/results/*/results-prefill-warm{1..5}.json`. Prefill was not the target of V1; the gain comes from the same kernel fusions.

### Decode at concurrency 1 / 2 / 4 / 8 (short prompts, 1,024 output tokens)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/decode-aggregate-dark.svg">
  <img alt="Aggregate decode throughput vs concurrency, V1 solid and baseline dashed: V1 prose 92, 161, 273, 416 tok/s; coding 120, 202, 328, 519; counting 142, 239, 394, 610 at C1, C2, C4, C8" src="docs/charts/decode-aggregate-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/decode-per-user-dark.svg">
  <img alt="Per-user decode speed vs concurrency, V1 solid and baseline dashed: V1 prose 92, 80, 68, 52 tok/s; coding 120, 103, 83, 66; counting 142, 121, 99, 77 at C1, C2, C4, C8" src="docs/charts/decode-per-user-light.svg">
</picture>

| Workload | C | Baseline tok/s | V1 tok/s | Change | Per-user tok/s | TTFT p50 (s) |
|---|---|---|---|---|---|---|
| Prose (short story) | 1 | 75.3 | **92.1** | +22% | 92.1 | 0.14 |
|  | 2 | 123.1 | **160.9** | +31% | 80.3 | 0.56 |
|  | 4 | 193.5 | **273.1** | +41% | 67.8 | 0.44 |
|  | 8 | 286.7 | **416.0** | +45% | 52.1 | 0.93 |
| Coding (Python + tests) | 1 | 103.1 | **120.2** | +17% | 120.2 | 0.15 |
|  | 2 | 157.2 | **201.7** | +28% | 102.9 | 0.25 |
|  | 4 | 243.0 | **328.0** | +35% | 83.3 | 0.57 |
|  | 8 | 371.6 | **519.1** | +40% | 65.9 | 0.62 |
| Counting (1…400 in words) | 1 | 132.4 | **141.7** | +7% | 141.7 | 0.16 |
|  | 2 | 224.3 | **239.2** | +7% | 120.9 | 0.21 |
|  | 4 | 312.5 | **393.5** | +26% | 99.2 | 0.50 |
|  | 8 | 439.4 | **610.4** | +39% | 77.2 | 0.70 |

Medians of 2 rounds per cell. Solid lines in the charts are V1 and dashed lines are the baseline. Across the whole V1 benchmark the FP8 drafter's acceptance was **38.6% per draft token**, about **2.16 tokens per decode step** with 3 drafted.

### Decode at context 8K → 128K (prose task after a unique N-token document, 512 output tokens)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/decode-context-dark.svg">
  <img alt="Aggregate decode throughput at long context (V1): all five context lengths from 8K to 128K track each other, about 92-96 tok/s at C1 and 396-434 tok/s at C8" src="docs/charts/decode-context-light.svg">
</picture>

| Context | C | Baseline tok/s | V1 tok/s | Change | Per-user tok/s | TTFT p50 (s) |
|---|---|---|---|---|---|---|
| 8K | 1 | 79.3 | **96.1** | +21% | 96.1 | 1.6 |
| 8K | 2 | 125.6 | **150.6** | +20% | 68.0 | 2.3 |
| 8K | 4 | 197.8 | **265.6** | +34% | 56.7 | 5.4 |
| 8K | 8 | 302.9 | **433.6** | +43% | 36.8 | 8.4 |
| 16K | 1 | 82.1 | **92.3** | +13% | 92.3 | 3.0 |
| 16K | 2 | 123.6 | **167.4** | +35% | 76.1 | 5.3 |
| 16K | 4 | 193.0 | **266.3** | +38% | 49.5 | 9.0 |
| 16K | 8 | 307.3 | **427.0** | +39% | 28.4 | 15.0 |
| 32K | 1 | 84.5 | **91.6** | +8% | 91.6 | 6.0 |
| 32K | 2 | 129.8 | **155.6** | +20% | 59.2 | 9.0 |
| 32K | 4 | 192.0 | **261.6** | +36% | 31.9 | 15.0 |
| 32K | 8 | 288.0 | **404.7** | +41% | 17.3 | 27.8 |
| 64K | 1 | 65.5 | **93.8** | +43% | 93.8 | 12.7 |
| 64K | 2 | 119.2 | **151.3** | +27% | 51.6 | 18.6 |
| 64K | 4 | 207.3 | **268.5** | +29% | 20.2 | 30.5 |
| 64K | 8 | 290.8 | **424.0** | +46% | 9.9 | 55.1 |
| 128K | 1 | 73.9 | **91.5** | +24% | 91.5 | 26.1 |
| 128K | 2 | 121.0 | **154.6** | +28% | 45.4 | 38.4 |
| 128K | 4 | 176.5 | **247.5** | +40% | 11.8 | 64.2 |
| 128K | 8 | 290.3 | **395.8** | +36% | 5.2 | 116.1 |

Thanks to sparse attention, aggregate decode speed barely depends on context: about 396–434 tok/s at C8 from 8K to 128K. Per-user tok/s at higher C is much lower than aggregate ÷ C, because each request's decode overlaps the other requests' prefills. TTFT at high C is dominated by those queued prefills: 8 × 128K is about 1M prompt tokens.

### Power (450 W limit per GPU, 1 Hz `nvidia-smi` samples)

| Phase | GPU 0 avg / max | GPU 1 avg / max | Peak temp (GPU 0 / 1) |
|---|---|---|---|
| Prefill sweeps (5×, warm) | 392 / 436 W | 400 / 444 W | 69 / 81 °C |
| Decode, C1–C8 | 330 / 427 W | 345 / 429 W | 67 / 78 °C |
| Long-context decode | 405 / 432 W | 408 / 453 W | 69 / 82 °C |

V1 run. Idle is about 20–30 W per GPU with the model loaded. GPU 1 is the top card and runs about 12 °C hotter. The baseline's power log is in `bench/results/baseline/`.

### Reproduce

```bash
cd bench
POOL=<KV pool tokens from the server log> OUT=results/myrun ./run_all.sh   # warm-up, 5 prefill sweeps, decode matrix, context sweep, GPU power log
RESULTS=myrun python3 make_charts.py   # regenerates docs/charts/*.svg; draws results/baseline dashed
```

## Optimization targets

After V1 a decode step is about 20 ms and still about 8,000 kernel launches. Candidates for V2, all lossless:
- **Rotary embedding:** about 36 tiny kernels per layer (cos/sin lookup, compares, multiply/add, stack). The position-only part can be computed once per pass, and the rotation fused into one kernel.
- **LM head:** it runs as 5 separate full-matrix FP32 passes per step, about 1.2 ms. A single batched GEMM should cost about a quarter of that.
- **Small dense BF16 GEMMs:** about 5 ms per step in ~400 cuBLAS calls of a few tokens each (shared experts, latent projections, `q_a`/`kv` projections). Look at batching or fusing them with their neighbours.
- **Hyper-connection and gating glue:** the remaining elementwise ops around iHC and the gated-MLA sigmoid/multiply.
- **Sparse-MLA autotune:** there's no FlashInfer SM120 sparse-MLA autotune cache, so the kernels run with default tactics.
- **Verification cost at batch:** the step grows from 20 ms at 1 user to 36 ms at 8 users for 32 tokens. Find which components scale with tokens (MoE at 16 experts per token, per-token quantization) and batch them better.
- **Acceptance:** the first draft token is right 58% of the time. The drafter was trained against the BF16 model and here reads an FP8 KV cache from a 4-bit target. To separate those effects, measure the same drafter on the FP8 target.
- **Engram over PCIe:** 62 GiB of pinned tables are read through UVA every step. With the VRAM freed up, try moving some or all of them onto the GPUs.
- **Power:** run the same sweep at 500–600 W to see how much is power-bound.

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

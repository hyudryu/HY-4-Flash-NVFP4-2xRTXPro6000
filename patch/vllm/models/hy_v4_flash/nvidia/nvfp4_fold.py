# SPDX-License-Identifier: Apache-2.0
"""Local NVFP4 patch: give each expert's gate and up halves one global scale.

vLLM's NVFP4 MoE kernels keep a single w13 global scale per expert
(`w13_weight_scale_2[:, 0]`) and ignore the up-half value. The
Avifenesh/Hy4-flash-NVFP4 checkpoint stores separate per-expert global scales
for gate and up; they differ (by exactly 2x) for 182 of 20,992 experts.

The up half is re-encoded so that it dequantizes to identical values under the
gate's global scale. Per 16-element block, the factor 2**k is split between the
E4M3 block scale and the E2M1 codes (scale * 2**a, codes * 2**b, a + b = k),
choosing any split that is exactly representable. The SwiGLU clamp on `up`
(swiglu_limit) rules out compensating through w2 instead.

Blocks with no exact split (6 experts of layer 1 in this checkpoint: subnormal
E4M3 block scales that cannot be halved) fall back to rounding the scaled
block scale; this is logged. Measured on those experts: ~2e-6 relative
Frobenius error of the up half, vs ~6.6e-3 for the NVFP4 encoding itself.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# E2M1 magnitude index (0..7: 0, .5, 1, 1.5, 2, 3, 4, 6) after x2 / x0.5.
_UP = (0, 2, 4, 5, 6, 7, -1, -1)
_DOWN = (0, -1, 1, -1, 2, 3, 4, 5)
_FP8 = torch.float8_e4m3fn


def _rescale_codes(codes: torch.Tensor, steps: int):
    """Scale packed E2M1 nibbles by 2**steps. Returns (codes, representable)."""
    table = torch.tensor(_UP if steps > 0 else _DOWN, device=codes.device)
    out, ok = [], torch.ones_like(codes, dtype=torch.bool)
    for nibble in ((codes & 0xF).long(), (codes >> 4).long()):
        sign, mag = nibble & 8, nibble & 7
        for _ in range(abs(steps)):
            mag = torch.where(mag >= 0, table[mag.clamp_min(0)], mag)
        ok &= mag >= 0
        out.append((sign | mag.clamp_min(0)) & 0xF)
    return (out[0] | (out[1] << 4)).to(torch.uint8), ok


def refold(codes: torch.Tensor, scales: torch.Tensor, factor: float):
    """Re-encode NVFP4 (codes [..., in/2] u8, scales [..., in/16] e4m3) so it
    dequantizes to `factor` (a power of two) times the original values.

    Returns (codes, scales, rounded_blocks); rounded_blocks == 0 means exact.
    """
    k = round(torch.log2(torch.tensor(float(factor))).item())
    if 2.0**k != factor:
        raise ValueError(f"NVFP4 global-scale ratio {factor} is not a power of two")
    s = scales.float()
    blocks = codes.reshape(*codes.shape[:-1], -1, 8)  # 8 bytes = 16 values
    out_s = scales.view(torch.uint8).clone()
    out_c = blocks.clone()
    done = torch.zeros(s.shape, dtype=torch.bool, device=s.device)
    for b in (0, 1, -1, 2, -2, 3, -3):
        target = s * 2.0 ** (k - b)
        new_s = target.to(_FP8)
        use = (new_s.float() == target) & ~done
        if b:
            new_c, ok = _rescale_codes(blocks, b)
            use &= ok.all(-1)
        else:
            new_c = blocks
        if use.any():
            out_s[use] = new_s.view(torch.uint8)[use]
            out_c[use] = new_c[use]
            done |= use
        if bool(done.all()):
            break
    rest = ~done
    rounded = int(rest.sum())
    if rounded:
        out_s[rest] = (s * 2.0**k).to(_FP8).view(torch.uint8)[rest]
    return out_c.reshape(codes.shape), out_s.view(_FP8), rounded


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1, device=codes.device)
    vals = []
    for nibble in ((codes & 0xF).long(), (codes >> 4).long()):
        v = table[nibble & 7]
        vals.append(torch.where((nibble & 8) != 0, -v, v))
    v = torch.stack(vals, -1).flatten(-2)
    return v.view(*scales.shape, 16) * scales.float()[..., None]


def _rel_err(codes, scales, new_codes, new_scales, factor) -> float:
    ref = _dequant(codes, scales) * factor
    return ((_dequant(new_codes, new_scales) - ref).norm() / ref.norm()).item()


@torch.no_grad()
def unify_w13_global_scales(layer, layer_name: str) -> None:
    """Make w13_weight_scale_2[:, 0] == [:, 1] by re-encoding one half.

    Folds the up half into the gate's global scale, or the gate half into the
    up's, whichever is exact; if neither is, the one with the lower error.
    """
    s2 = layer.w13_weight_scale_2.data
    if s2.ndim != 2 or s2.shape[1] != 2:
        return
    mismatch = (s2[:, 0] != s2[:, 1]).nonzero().flatten().tolist()
    if not mismatch:
        return
    w, ws = layer.w13_weight.data, layer.w13_weight_scale.data
    half = w.shape[1] // 2  # rows [0, half) gate, [half, 2*half) up (local shard)
    gate, up = slice(0, half), slice(half, 2 * half)
    rounded_total, rounded_experts, worst = 0, 0, 0.0
    for e in mismatch:
        # (rows to re-encode, factor, global-scale column that is kept)
        options = (
            (up, (s2[e, 1] / s2[e, 0]).item(), 0),
            (gate, (s2[e, 0] / s2[e, 1]).item(), 1),
        )
        best = None
        for rows, factor, keep in options:
            codes, scales, rounded = refold(w[e, rows], ws[e, rows], factor)
            err = (
                0.0
                if rounded == 0
                else _rel_err(w[e, rows], ws[e, rows], codes, scales, factor)
            )
            if best is None or err < best[0]:
                best = (err, rows, keep, codes, scales, rounded)
            if rounded == 0:
                break
        err, rows, keep, codes, scales, rounded = best
        w[e, rows] = codes
        ws[e, rows] = scales
        s2[e, 1 - keep] = s2[e, keep]
        rounded_total += rounded
        rounded_experts += bool(rounded)
        worst = max(worst, err)
    if rounded_total:
        logger.warning(
            "%s: %d experts had gate/up NVFP4 global scales that differ; %d of "
            "them needed rounding of %d block scales (all others exact); worst "
            "relative error of a re-encoded half: %.2e.",
            layer_name, len(mismatch), rounded_experts, rounded_total, worst,
        )
    else:
        logger.info(
            "%s: unified gate/up NVFP4 global scales for %d experts (exact).",
            layer_name, len(mismatch),
        )

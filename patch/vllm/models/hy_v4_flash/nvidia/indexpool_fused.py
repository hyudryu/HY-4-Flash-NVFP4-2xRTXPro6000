# SPDX-License-Identifier: Apache-2.0
"""V2: fused IndexPool glue for HY4 Flash (local overlay, not upstream).

`indexpool_ops.rotate_indexer` + `fwht_quantize` run as ~90 one-microsecond
torch kernels per call, twice per sparse layer, and `expand_groups` as ~35.
The kernels here compute the same values in one launch each:

- `rotate_fwht_quantize`: RoPE rotation (fp32, rounded to the input dtype),
  the 7-stage Walsh-Hadamard butterfly in the same pairing order, the
  power-of-two scale, and the FP8 E4M3 cast (round-to-nearest-even).
- `expand_groups_into`: group ids -> token indices with stable compaction,
  written straight into the (already -1 filled) top-k buffer.

`tests/` in the repo checks both against the reference implementations.
"""

import torch

from vllm.triton_utils import tl, triton

INDEX_DIM = 128
FP8_MAX = 448.0


@triton.jit
def _butterfly(x, S: tl.constexpr):
    y = tl.permute(tl.reshape(x, (128 // (2 * S), 2, S)), (0, 2, 1))
    left, right = tl.split(y)
    z = tl.permute(tl.join(left + right, left - right), (0, 2, 1))
    return tl.reshape(z, (128,))


@triton.jit
def _rotate_fwht_quant(
    X,
    POS,
    INV,
    Q,
    SCALE,
    ZERO,
    X_ROW,
    POS_PLANE,
    POS_TOK,
    Q_PITCH,
    RPP: tl.constexpr,
    ROPE: tl.constexpr,
    SEC_H: tl.constexpr,
    SEC_W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, 128)
    nope = 128 - ROPE
    half = ROPE // 2
    j = tl.maximum(k - nope, 0)
    i = j % half
    plain = tl.load(X + row * X_ROW + k).to(tl.float32)
    # rotate_indexer de-interleaves the rotary block: evens, then odds.
    even = tl.load(X + row * X_ROW + nope + 2 * i).to(tl.float32)
    odd = tl.load(X + row * X_ROW + nope + 2 * i + 1).to(tl.float32)
    axis = tl.where(
        (i % 3 == 1) & (i < SEC_H * 3), 1, tl.where((i % 3 == 2) & (i < SEC_W * 3), 2, 0)
    )
    token = row // RPP
    pos = tl.load(POS + axis * POS_PLANE + token * POS_TOK).to(tl.float32)
    freq = pos * tl.load(INV + i)
    cos = tl.cos(freq)
    sin = tl.sin(freq)
    # The reference rounds each product to float32 before adding. Triton would
    # contract `a * b + c * d` into an FMA (1-ulp differences), so pass each
    # product through an integer add of a zero the compiler cannot see.
    zero = tl.load(ZERO)
    ec = ((even * cos).to(tl.int32, bitcast=True) + zero).to(tl.float32, bitcast=True)
    os_ = ((odd * sin).to(tl.int32, bitcast=True) + zero).to(tl.float32, bitcast=True)
    oc = ((odd * cos).to(tl.int32, bitcast=True) + zero).to(tl.float32, bitcast=True)
    es = ((even * sin).to(tl.int32, bitcast=True) + zero).to(tl.float32, bitcast=True)
    rot = tl.where(j >= half, oc + es, ec - os_)
    # `.to(x.dtype)` in the reference. Triton's own float32 -> bfloat16 cast
    # does not round ties to even, so round the bit pattern explicitly.
    rb = rot.to(tl.uint32, bitcast=True)
    low = rb & 0xFFFF
    high = rb >> 16
    high = high + ((low > 0x8000) | ((low == 0x8000) & ((high & 1) == 1))).to(tl.uint32)
    rot = (high << 16).to(tl.float32, bitcast=True)
    x = tl.where(k >= nope, rot, plain)
    # Walsh-Hadamard butterflies, same stage order and pairing as fwht_quantize.
    x = _butterfly(x, 1)
    x = _butterfly(x, 2)
    x = _butterfly(x, 4)
    x = _butterfly(x, 8)
    x = _butterfly(x, 16)
    x = _butterfly(x, 32)
    x = _butterfly(x, 64)
    x = x * 0.08838834764831845
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1e-4) / 448.0)))
    v = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    # float32 -> float8_e4m3fn, round to nearest even. |v| <= 448 < max.
    bits = v.to(tl.int32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    exp = (mag >> 23) - 127
    man = mag & 0x7FFFFF
    rem = man & 0xFFFFF
    m3 = man >> 20
    up = (rem > 0x80000) | ((rem == 0x80000) & ((m3 & 1) == 1))
    normal = ((exp + 7) << 3) + m3 + up.to(tl.int32)
    sub = tl.abs(v) * 512.0
    floor = tl.floor(sub)
    frac = sub - floor
    fi = floor.to(tl.int32)
    sub_code = fi + ((frac > 0.5) | ((frac == 0.5) & ((fi & 1) == 1))).to(tl.int32)
    code = tl.where(exp >= -6, normal, sub_code)
    code = tl.where(bits < 0, code | 0x80, code)
    tl.store(Q + token * Q_PITCH + (row % RPP) * 128 + k, code.to(tl.uint8))
    tl.store(SCALE + row, scale)


_inverse_cache: dict = {}
_zero_cache: dict = {}


def _zero(device) -> torch.Tensor:
    zero = _zero_cache.get(device)
    if zero is None:
        zero = _zero_cache[device] = torch.zeros(1, dtype=torch.int32, device=device)
    return zero


def _inverse_frequencies(rope_dim: int, theta: float, device) -> torch.Tensor:
    key = (rope_dim, float(theta), device)
    inverse = _inverse_cache.get(key)
    if inverse is None:
        # Same expression as rotate_indexer, evaluated once.
        inverse = theta ** (
            -torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
        _inverse_cache[key] = inverse
    return inverse


def rotate_fwht_quantize(x, positions, rope_dim, parameters, rows_per_position=1,
                         pad_rows=None):
    """`fwht_quantize(rotate_indexer(x, positions).reshape(-1, 128))` in one launch.

    `x` is `[positions * rows_per_position, 128]` (strided rows allowed).
    Returns `(quant uint8 [positions, pad_rows or rows_per_position, 128],
    scale float32 [rows])`; rows beyond `rows_per_position` are zero.
    """
    if x.dtype != torch.bfloat16:
        raise ValueError("fused IndexPool rotation requires bfloat16 vectors")
    if x.ndim != 2 or x.shape[1] != INDEX_DIM or x.stride(1) != 1:
        raise ValueError("IndexPool quantization requires [rows, 128] vectors")
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Flash IndexPool currently requires default RoPE")
    if rope_dim <= 0 or rope_dim > INDEX_DIM or rope_dim % 4:
        raise ValueError("fused IndexPool rotation requires 0 < rope_dim <= 128")
    rows = x.shape[0]
    count = rows // rows_per_position
    if count * rows_per_position != rows or positions.shape[-1] != count:
        raise ValueError("IndexPool rotation rows do not match positions")
    section_h = section_w = 0
    if positions.ndim == 2:
        section = parameters.get("mrope_section")
        if positions.shape[0] != 3 or section is None or sum(section) * 2 != rope_dim:
            raise ValueError("IndexPool requires valid three-axis MRoPE sections")
        if not parameters.get("mrope_interleaved", True):
            raise ValueError("fused IndexPool rotation requires interleaved MRoPE")
        section_h, section_w = section[1], section[2]
    heads = pad_rows or rows_per_position
    if heads == rows_per_position:
        quant = torch.empty((count, heads, INDEX_DIM), dtype=torch.uint8, device=x.device)
    else:
        quant = torch.zeros((count, heads, INDEX_DIM), dtype=torch.uint8, device=x.device)
    scale = torch.empty(rows, dtype=torch.float32, device=x.device)
    if rows:
        inverse = _inverse_frequencies(rope_dim, parameters.get("rope_theta", 10000.0), x.device)
        _rotate_fwht_quant[(rows,)](
            x,
            positions,
            inverse,
            quant,
            scale,
            _zero(x.device),
            x.stride(0),
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.stride(-1),
            heads * INDEX_DIM,
            rows_per_position,
            rope_dim,
            section_h,
            section_w,
            num_warps=4,
        )
    return quant, scale


@triton.jit
def _expand_into(
    G,
    LOGICAL,
    LIVE,
    OUT,
    G_ROW,
    OUT_ROW,
    BUDGET: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    g = j // RATIO
    off = (j % RATIO).to(tl.int64)
    pos = tl.load(LOGICAL + row)
    live = tl.load(LIVE + row).to(tl.int32) != 0
    live = live & (pos >= 0)
    upper = tl.maximum(pos, 0) // RATIO
    is_hist = g < BUDGET
    gid = tl.load(G + row * G_ROW + g, mask=is_hist, other=-1).to(tl.int64)
    tail = upper * RATIO + off
    valid = tl.where(
        is_hist,
        (gid >= 0) & (gid < upper) & live,
        (g == BUDGET) & live & (tail <= pos),
    )
    candidate = tl.where(is_hist, gid * RATIO + off, tail)
    dest = tl.cumsum(valid.to(tl.int32), 0) - 1
    tl.store(OUT + row * OUT_ROW + dest, candidate.to(tl.int32), mask=valid)


def expand_groups_into(output, group_ids, positions, ratio, active):
    """`output.copy_(expand_groups(group_ids, positions, ratio, active))` for an
    `output` that is already filled with -1."""
    rows, budget = group_ids.shape
    width = (budget + 1) * ratio
    if (
        output.shape != (rows, width)
        or output.dtype != torch.int32
        or group_ids.dtype != torch.int32
        or positions.dtype != torch.int64
        or active.dtype != torch.bool
        or output.stride(1) != 1
        or group_ids.stride(1) != 1
    ):
        raise ValueError("unexpected IndexPool expansion layout")
    if rows:
        _expand_into[(rows,)](
            group_ids,
            positions,
            active,
            output,
            group_ids.stride(0),
            output.stride(0),
            budget,
            ratio,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    return output

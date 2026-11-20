"""Bit-exactness of the fused IndexPool kernels against the reference torch code.
Run inside the vLLM image with the repo mounted: python3 tests/test_indexpool_fused.py"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patch", "vllm", "models", "hy_v4_flash", "nvidia"))
from vllm.models.hy_v4_flash.nvidia.indexpool_ops import rotate_indexer, fwht_quantize, expand_groups
import indexpool_fused as F2
dev = "cuda"; torch.manual_seed(0)
P = {"rope_type": "default", "rope_theta": 10000000, "mrope_section": [12, 10, 10], "mrope_interleaved": True}
bad = 0
def check(name, a, b):
    global bad
    n = (a != b).sum().item()
    print(f"{name:55s} {'OK' if n == 0 else f'MISMATCH {n}/{a.numel()}'}"); bad += n
for trial, (rows_tok, heads, scale_mag, maxpos) in enumerate([(4,16,1.0,1000),(32,16,3.0,500000),(257,16,0.01,524287),(1,16,50.0,7),(64,1,1.0,130000),(3,1,1e-6,5),(500,1,20.0,524287)]):
    x = (torch.randn(rows_tok*heads, 128, device=dev) * scale_mag).bfloat16()
    if trial == 2: x[::7] = 0
    for nd in (2, 1):
        pos = torch.randint(0, maxpos+1, (3, rows_tok), device=dev) if nd == 2 else torch.randint(0, maxpos+1, (rows_tok,), device=dev)
        if nd == 2 and trial % 2 == 0: pos[1:] = pos[0]          # text: all three axes equal
        xr = x.reshape(rows_tok, heads, 128) if heads > 1 else x
        ref = rotate_indexer(xr, pos, 64, P)
        rq, rs = fwht_quantize(ref.reshape(-1, 128).bfloat16())
        q, s = F2.rotate_fwht_quantize(x, pos, 64, P, rows_per_position=heads, pad_rows=32 if heads > 1 else None)
        qq = q[:, :heads].reshape(-1, 128)
        check(f"trial {trial} nd={nd} quant bytes ({rows_tok}x{heads})", qq, rq.view(torch.uint8))
        check(f"trial {trial} nd={nd} scales", s.view(torch.int32), rs.view(torch.int32))
        if heads > 1: check(f"trial {trial} nd={nd} pad zero", q[:, heads:], torch.zeros_like(q[:, heads:]))
# FP8 cast sweep: reuse kernel path is hard to isolate; test the cast logic through extreme inputs instead.
for mag in (1e-8, 1e-3, 0.3, 7.0, 400.0, 3e4):
    x = (torch.randn(4096, 128, device=dev) * mag).bfloat16(); pos = torch.zeros(4096, dtype=torch.long, device=dev)
    rq, rs = fwht_quantize(rotate_indexer(x, pos, 64, P)); q, s = F2.rotate_fwht_quantize(x, pos, 64, P)
    check(f"cast sweep mag={mag:g} bytes", q.reshape(-1,128), rq.view(torch.uint8)); check(f"cast sweep mag={mag:g} scales", s.view(torch.int32), rs.view(torch.int32))
# expand_groups
for trial in range(6):
    n = [1, 4, 32, 7, 100, 3][trial]; budget, ratio = 511, 4
    logical = torch.randint(0, [5, 600, 5000, 524287, 100000, 2][trial] + 1, (n,), device=dev)
    if trial == 3: logical[0] = -1
    live = torch.rand(n, device=dev) > (0.2 if trial % 2 else 0.0)
    upper = logical.clamp_min(0) // ratio
    g = torch.full((n, budget), -1, dtype=torch.int32, device=dev)
    for r in range(n):
        k = min(budget, int(upper[r])); 
        if k:
            ids = torch.randperm(int(upper[r]), device=dev)[:k].sort().values.int(); g[r, :k] = ids
    if trial == 4: g[0, 5] = -1; g[1, 3] = int(upper[1]) + 5      # unsorted / out-of-range entries
    ref = expand_groups(g, logical, ratio, active=live)
    out = torch.full_like(ref, -1); F2.expand_groups_into(out, g, logical, ratio, live)
    check(f"expand trial {trial} rows={n}", out, ref)
print("TOTAL MISMATCHES", bad)

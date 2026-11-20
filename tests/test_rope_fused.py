"""Fused M-RoPE path (per-pass token rows + vLLM's C rotary kernel) vs the original
MRotaryEmbedding.forward_native, both scored against an exact float64 reference.
The two are not bit-identical: the original rounds to BF16 three times per element,
the fused kernel once. Run inside the vLLM image: python3 tests/test_rope_fused.py"""
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.hy_v4.nvidia.mrope import apply_mrope, set_token_rope
dev="cuda"; torch.manual_seed(0)
with set_current_vllm_config(VllmConfig()):
    rope = get_rope(64, max_position=524288, is_neox_style=False, rope_parameters={"rope_theta": 10000000, "rope_type": "default", "mrope_section": [12, 10, 10], "mrope_interleaved": True}).to(dev)
T,H=2048,32
pos = torch.randint(0, 524288, (T,), device=dev).expand(3, T).contiguous()
q = torch.randn(T, H, 64, device=dev).bfloat16(); kv = torch.randn(T, 576, device=dev).bfloat16()
ref_q,_ = rope.forward_native(pos, q.reshape(T,-1), kv[:,512:])
q2=q.clone(); kv2=kv.clone(); set_token_rope(rope,pos,q2); rows=rope._hy4_token_rope[1]
apply_mrope(rope,pos,q2,kv2[:,512:].unsqueeze(1))
cos,sin=rows.double().chunk(2,-1)                      # the bf16 cos/sin both paths read
x=q.double(); x1,x2=x[...,::2],x[...,1::2]
exact=torch.stack((x1*cos[:,None]-x2*sin[:,None], x2*cos[:,None]+x1*sin[:,None]),-1).flatten(-2)
best=exact.to(torch.bfloat16)                          # correctly rounded result
print("native == correctly rounded: %.1f%%"%((ref_q.reshape(T,H,64)==best).float().mean().item()*100))
print("fused  == correctly rounded: %.1f%%"%((q2==best).float().mean().item()*100))
print("native == fused: %.1f%%"%((ref_q.reshape(T,H,64)==q2).float().mean().item()*100))
print("mean abs error native %.3g | fused %.3g"%((ref_q.reshape(T,H,64).double()-exact).abs().mean().item(), (q2.double()-exact).abs().mean().item()))

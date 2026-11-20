# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 (RTX PRO 6000 / RTX 50 / GB10) sparse MLA backends for HY4 Flash.

Local overlay patch, same approach as vllm-project/vllm#68387:
FlashMLA only ships SM90/SM100 kernels, so on SM12x HY4 Flash runs on
FlashInfer's SM120 sparse MLA kernel, which takes the per-head learnable sink
natively (`sinks=`); vLLM's stock FlashInferMLASparseSM120Impl drops it.

Differences from the stock SM120 impl:
- forwards the learnable sink to the kernel;
- accepts `sliding_window` for HY4's SWA layers. The window is enforced by
  `FlashWindowIndexer`, which writes causal window indices into the shared
  top-k buffer, so the kernel itself stays window-agnostic;
- reads fp8_ds_mla tile scales as `arbitrary_fp32` (verified against the
  bytes vLLM's concat_and_cache_mla writes; `pow2_fp32` gave ~3x the error,
  see vllm#54563).

`get_name()` is inherited ("FLASHINFER_MLA_SPARSE_SM120") on purpose:
`_canonicalize_sparse_mla_kv_cache_dtype` keys on it to map "auto" to the
fp8_ds_mla layout this kernel requires.
"""

import torch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)


class FlashSM120Impl(FlashInferMLASparseSM120Impl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        indexer=None,
        **mla_args,
    ) -> None:
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            None,  # window handled by FlashWindowIndexer's top-k indices
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            **mla_args,
        )
        if sinks is not None and (
            sinks.dtype != torch.float32 or sinks.shape != (num_heads,)
        ):
            raise ValueError(
                f"HY4 SM120 sinks must be float32 ({num_heads},); got "
                f"{sinks.dtype} {tuple(sinks.shape)}"
            )
        self.sinks = sinks
        self.kv_scale_format = "arbitrary_fp32"

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices_physical: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        output = q.new_empty((q.shape[0], self.num_heads, self.kv_lora_rank))
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        sparse_capacity = topk_indices_physical.shape[1]
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=sparse_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_capacity,
            kv_scale_format=self.kv_scale_format,
            sinks=None if self.sinks is None else [self.sinks],
        )
        return out.squeeze(1)


class FlashSM120SparseBackend(FlashInferMLASparseSM120Backend):
    @staticmethod
    def get_impl_cls():
        return FlashSM120Impl

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False


class FlashSM120SWABackend(FlashSM120SparseBackend):
    @classmethod
    def supports_sliding_window(cls):
        return True

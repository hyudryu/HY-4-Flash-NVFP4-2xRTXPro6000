# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal window indexing over engine-managed sliding MLA pages."""

from typing import Any

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseBackend
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadataBuilder


logger = init_logger(__name__)


class FlashSparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The direct paged kernel never gathers evicted SWA history.
        self.fp8_use_mixed_batch = True


class FlashSparseBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_builder_cls():
        return FlashSparseMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False


class FlashSWABackend(FlashSparseBackend):
    @classmethod
    def supports_sliding_window(cls):
        return True


def window_indices(positions: torch.Tensor, window: int, width: int) -> torch.Tensor:
    if window < 1 or width < window:
        raise ValueError("SWA index width must cover its causal window")
    columns = torch.arange(width, device=positions.device)
    indices = positions[:, None] - window + 1 + columns
    live = (positions[:, None] >= 0) & (columns < window) & (indices >= 0)
    return torch.where(live, indices, -1).int()


class FlashWindowIndexer(nn.Module):
    # V1: per-process cache of the current forward pass's window indices.
    _shared: dict = {}

    def __init__(self, window: int, topk_indices_buffer, attention_prefix: str):
        super().__init__()
        self.window = window
        self.topk_indices_buffer = topk_indices_buffer
        self.attention_prefix = attention_prefix
        if topk_indices_buffer.shape[1] < window:
            raise ValueError("SWA index width must cover its causal window")

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("Flash SWA requires per-layer metadata")
        meta: Any = metadata[self.attention_prefix]
        if meta.num_actual_tokens == 0 or meta.seq_lens.numel() == 0:
            return output.fill_(-1)
        # V1: the window indices depend only on the batch metadata, not on the
        # layer, so every SWA layer of one forward pass computes the same
        # tensor (~30 tiny kernels each). Compute once per forward and copy.
        # Keyed on the forward-context object (new per forward) and the layer
        # metadata object, so a different pass or cache group never matches.
        cache = FlashWindowIndexer._shared
        context = get_forward_context()
        key = (
            self.window,
            output.shape[0],
            output.shape[1],
            output.device,
            meta.num_actual_tokens,
        )
        # The result is a pure function of (query_start_loc, seq_lens,
        # num_actual_tokens, window, width); match on the tensor objects.
        hit = (
            cache.get("context") is context
            and cache.get("starts") is meta.query_start_loc
            and cache.get("seq_lens") is meta.seq_lens
            and cache.get("key") == key
        )
        if not cache.get("logged") and cache.get("context") is context:
            cache["logged"] = True
            logger.info(
                "Flash SWA window-index cache: hit=%s (starts same=%s, seq_lens "
                "same=%s, key same=%s)",
                hit,
                cache.get("starts") is meta.query_start_loc,
                cache.get("seq_lens") is meta.seq_lens,
                cache.get("key") == key,
            )
        if hit:
            return output.copy_(cache["buffer"][: output.shape[0]])
        rows = torch.arange(output.shape[0], device=output.device)
        starts = meta.query_start_loc
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = meta.seq_lens[requests] - (starts[requests + 1] - starts[requests])
        logical = logical + rows - starts[requests]
        live = (rows < meta.num_actual_tokens) & (rows < starts[-1])
        output.copy_(
            window_indices(torch.where(live, logical, -1), self.window, output.shape[1])
        )
        # One persistent full-size buffer (never allocated inside a CUDA graph
        # capture after the first use); rows beyond this batch are not read.
        buffer = cache.get("buffer")
        full = self.topk_indices_buffer
        if buffer is None or buffer.shape != full.shape or buffer.device != full.device:
            buffer = torch.empty_like(full)
        buffer[: output.shape[0]].copy_(output)
        cache.update(
            context=context,
            starts=meta.query_start_loc,
            seq_lens=meta.seq_lens,
            key=key,
            buffer=buffer,
        )
        return output

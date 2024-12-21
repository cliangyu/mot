# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

from lingua import probe

flex_attention_comp = torch.compile(flex_attention)


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024
    
    # MoT specific args
    use_mot: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text", "image"])
    attention_type: str = "mot"  # Can be "mot" or "mst"


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_flattened(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to flattened query and key tensors.

    Args:
        xq: Query tensor of shape [N, num_heads, head_dim].
        xk: Key tensor of shape [N, num_heads, head_dim].
        freqs_cis: Precomputed rotary embeddings of shape [max_seqlen, head_dim // 2, 2, 2].
        positions: Token positions tensor of shape [N].

    Returns:
        Tuple of transformed xq and xk tensors.
    """
    N, num_heads, head_dim = xq.shape
    head_dim_half = head_dim // 2

    # Reshape xq and xk to [N, num_heads, head_dim_half, 2]
    xq_ = xq.view(N, num_heads, head_dim_half, 2)
    xk_ = xk.view(N, num_heads, head_dim_half, 2)

    # Get freqs_cis for the positions
    freqs_cis_pos = freqs_cis[positions]  # [N, head_dim_half, 2, 2]
    # Expand freqs_cis_pos to [N, 1, head_dim_half, 2, 2] to match xq_
    freqs_cis_pos = freqs_cis_pos.unsqueeze(1)  # [N, 1, head_dim_half, 2, 2]

    # Perform complex multiplication (rotary embedding application)
    xq_out = torch.einsum('n h d p, n h d p q -> n h d q', xq_, freqs_cis_pos)
    xk_out = torch.einsum('n h d p, n h d p q -> n h d q', xk_, freqs_cis_pos)

    # Flatten back to [N, num_heads, head_dim]
    xq_out = xq_out.reshape(N, num_heads, head_dim)
    xk_out = xk_out.reshape(N, num_heads, head_dim)

    return xq_out, xk_out


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class MoTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        modalities: List[str],
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta
        self.modalities = modalities
        self.heads_per_group = self.n_heads // self.n_kv_heads

        # Modality-specific Q, K, V projections
        self.wq_m = nn.ModuleDict({
            m: nn.Linear(dim, n_heads * head_dim, bias=False)
            for m in modalities
        })
        self.wk_m = nn.ModuleDict({
            m: nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            for m in modalities
        })
        self.wv_m = nn.ModuleDict({
            m: nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            for m in modalities
        })
        self.wo_m = nn.ModuleDict({
            m: nn.Linear(n_heads * head_dim, dim, bias=False)
            for m in modalities
        })

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        modality_ids: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        # Create modality masks
        modality_ids = modality_ids.view(bsz, seq_len)
        modality_id_to_name = {idx: m for idx, m in enumerate(self.modalities)}
        modality_masks = {
            m: (modality_ids == idx)
            for idx, m in modality_id_to_name.items()
        }

        # Initialize tensors for Q, K, V
        xq = torch.zeros(bsz, seq_len, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        xk = torch.zeros(bsz, seq_len, self.n_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
        xv = torch.zeros(bsz, seq_len, self.n_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)

        # Apply modality-specific projections
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                x_m = x[indices]
                out_q = self.wq_m[modality](x_m)
                out_k = self.wk_m[modality](x_m)
                out_v = self.wv_m[modality](x_m)

                # Reshape outputs
                out_q = out_q.view(-1, self.n_heads, self.head_dim)
                out_k = out_k.view(-1, self.n_kv_heads, self.head_dim)
                out_v = out_v.view(-1, self.n_kv_heads, self.head_dim)

                # Assign back to xq, xk, xv
                xq[indices[0], indices[1]] = out_q
                xk[indices[0], indices[1]] = out_k
                xv[indices[0], indices[1]] = out_v

        # Reshape for attention
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        # Repeat keys and values if necessary
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # Compute attention
        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()
        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported")

        # Reshape output and apply modality-specific output projections
        output = output.view(bsz, seq_len, -1)
        final_output = torch.zeros_like(output)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                out_m = output[indices]
                final_m = self.wo_m[modality](out_m)
                final_output[indices[0], indices[1]] = final_m

        return final_output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        # Initialize modality-specific projections
        for m in self.modalities:
            nn.init.trunc_normal_(
                self.wq_m[m].weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_(
                self.wk_m[m].weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_(
                self.wv_m[m].weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_(
                self.wo_m[m].weight,
                mean=0.0,
                std=init_std / factor,
                a=-3 * (init_std / factor),
                b=3 * (init_std / factor),
            )
        print(f"MoT Attention: Initialized parameters for modalities: {self.modalities} with factor {factor} and std {init_std}")


class MoTFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        modalities: List[str],
        mp_size: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.modalities = modalities

        # Modality-specific FFN layers
        self.w1_m = nn.ModuleDict({
            m: nn.Linear(dim, self.hidden_dim, bias=False)
            for m in modalities
        })
        self.w2_m = nn.ModuleDict({
            m: nn.Linear(self.hidden_dim, dim, bias=False)
            for m in modalities
        })
        self.w3_m = nn.ModuleDict({
            m: nn.Linear(dim, self.hidden_dim, bias=False)
            for m in modalities
        })

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        # Create modality masks
        modality_ids = modality_ids.view(bsz, seq_len)
        modality_id_to_name = {idx: m for idx, m in enumerate(self.modalities)}
        modality_masks = {
            m: (modality_ids == idx)
            for idx, m in modality_id_to_name.items()
        }

        output = torch.zeros_like(x)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                x_m = x[indices]
                x1 = self.w1_m[modality](x_m)
                x3 = self.w3_m[modality](x_m)
                out_m = self.w2_m[modality](F.silu(x1) * x3)
                output[indices[0], indices[1]] = out_m
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor

        for m in self.modalities:
            # Initialize w1_m and w3_m
            for w in [self.w1_m[m], self.w3_m[m]]:
                nn.init.trunc_normal_(
                    w.weight,
                    mean=0.0,
                    std=in_init_std,
                    a=-3 * in_init_std,
                    b=3 * in_init_std,
                )
            # Initialize w2_m
            nn.init.trunc_normal_(
                self.w2_m[m].weight,
                mean=0.0,
                std=out_init_std,
                a=-3 * out_init_std,
                b=3 * out_init_std,
            )
        print(f"MoT FFN: Initialized parameters for modalities: {self.modalities} with factor {factor} and std {init_std}")


class MoTTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.dim = args.dim
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // self.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        modalities = args.modalities or ['text', 'image']
        self.modalities = modalities

        # Modality-specific layer norms
        self.attention_norm_m = nn.ModuleDict({
            m: RMSNorm(args.dim, eps=args.norm_eps)
            for m in modalities
        })
        self.ffn_norm_m = nn.ModuleDict({
            m: RMSNorm(args.dim, eps=args.norm_eps)
            for m in modalities
        })

        # Replace attention and feed_forward with MoT versions
        self.attention = MoTAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            modalities=modalities,
        )
        self.feed_forward = MoTFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            modalities=modalities,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        modality_ids: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        # Create modality masks
        modality_ids = modality_ids.view(bsz, seq_len)
        modality_id_to_name = {idx: m for idx, m in enumerate(self.modalities)}
        modality_masks = {
            m: (modality_ids == idx)
            for idx, m in modality_id_to_name.items()
        }

        # Apply modality-specific attention layer norms
        x_norm = torch.zeros_like(x)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                x_m = x[indices]
                x_norm_m = self.attention_norm_m[modality](x_m)
                x_norm[indices[0], indices[1]] = x_norm_m

        # Apply attention
        h = x + self.attention(
            x_norm,
            freq_cis,
            modality_ids,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )

        # Apply modality-specific FFN layer norms
        h_norm = torch.zeros_like(h)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                h_m = h[indices]
                h_norm_m = self.ffn_norm_m[modality](h_m)
                h_norm[indices[0], indices[1]] = h_norm_m

        # Apply feed-forward network
        out = h + self.feed_forward(h_norm, modality_ids)
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        for norm in self.attention_norm_m.values():
            norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        for norm in self.ffn_norm_m.values():
            norm.reset_parameters()


def check_modality_coverage(modality_masks: dict, seq_len: int, device: torch.device, batch_size: int = 1) -> bool:
    """Check if modality masks cover the entire sequence exactly once (MECE rule).
    
    Args:
        modality_masks: Dict of modality name to boolean mask tensor
        seq_len: Expected sequence length
        device: Device of the tensors
        batch_size: Batch size for the masks
        
    Returns:
        bool: True if masks are mutually exclusive and collectively exhaustive
    """
    # Sum all masks to check if each position is covered exactly once
    total_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    
    # For each modality mask, we only need the sequence dimension mask
    # So we take [..., 0] to remove the model dimension
    for mask in modality_masks.values():
        # Remove the model dimension if it exists
        if mask.ndim == 3:  # If shape is [batch, seq_len, model_dim]
            mask = mask[..., 0]  # Take just [batch, seq_len]
            
        # Verify shapes match
        if mask.shape != total_mask.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match expected shape {total_mask.shape}")
            
        # Check no overlap with existing mask
        if (total_mask & mask).any():
            raise ValueError("Modality masks overlap - not mutually exclusive")
            
        total_mask |= mask
    
    # Check all positions are covered
    if not total_mask.all():
        raise ValueError("Modality masks don't cover all positions - not collectively exhaustive")
    
    return True

class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.use_mot = args.use_mot
        self.modalities = args.modalities
        self.attention_type = args.attention_type
        
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            if self.use_mot:
                if args.attention_type == "mst":
                    self.layers.append(MSTTransformerBlock(args))
                else:
                    self.layers.append(MoTTransformerBlock(args))
            else:
                self.layers.append(TransformerBlock(args))
    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        modality_ids: Optional[torch.Tensor] = None,
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl, modality_ids=modality_ids)
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)


# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

from lingua import probe

flex_attention_comp = torch.compile(flex_attention)


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024
    
    # MoT specific args
    use_mot: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text", "image"])
    attention_type: str = "mot"  # Can be "mot" or "mst"


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_flattened(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to flattened query and key tensors.

    Args:
        xq: Query tensor of shape [N, num_heads, head_dim].
        xk: Key tensor of shape [N, num_heads, head_dim].
        freqs_cis: Precomputed rotary embeddings of shape [max_seqlen, head_dim // 2, 2, 2].
        positions: Token positions tensor of shape [N].

    Returns:
        Tuple of transformed xq and xk tensors.
    """
    N, num_heads, head_dim = xq.shape
    head_dim_half = head_dim // 2

    # Reshape xq and xk to [N, num_heads, head_dim_half, 2]
    xq_ = xq.view(N, num_heads, head_dim_half, 2)
    xk_ = xk.view(N, num_heads, head_dim_half, 2)

    # Get freqs_cis for the positions
    freqs_cis_pos = freqs_cis[positions]  # [N, head_dim_half, 2, 2]
    # Expand freqs_cis_pos to [N, 1, head_dim_half, 2, 2] to match xq_
    freqs_cis_pos = freqs_cis_pos.unsqueeze(1)  # [N, 1, head_dim_half, 2, 2]

    # Perform complex multiplication (rotary embedding application)
    xq_out = torch.einsum('n h d p, n h d p q -> n h d q', xq_, freqs_cis_pos)
    xk_out = torch.einsum('n h d p, n h d p q -> n h d q', xk_, freqs_cis_pos)

    # Flatten back to [N, num_heads, head_dim]
    xq_out = xq_out.reshape(N, num_heads, head_dim)
    xk_out = xk_out.reshape(N, num_heads, head_dim)

    return xq_out, xk_out


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


def check_modality_coverage(modality_masks: dict, seq_len: int, device: torch.device, batch_size: int = 1) -> bool:
    """Check if modality masks cover the entire sequence exactly once (MECE rule).
    
    Args:
        modality_masks: Dict of modality name to boolean mask tensor
        seq_len: Expected sequence length
        device: Device of the tensors
        batch_size: Batch size for the masks
        
    Returns:
        bool: True if masks are mutually exclusive and collectively exhaustive
    """
    # Sum all masks to check if each position is covered exactly once
    total_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    
    # For each modality mask, we only need the sequence dimension mask
    # So we take [..., 0] to remove the model dimension
    for mask in modality_masks.values():
        # Remove the model dimension if it exists
        if mask.ndim == 3:  # If shape is [batch, seq_len, model_dim]
            mask = mask[..., 0]  # Take just [batch, seq_len]
            
        # Verify shapes match
        if mask.shape != total_mask.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match expected shape {total_mask.shape}")
            
        # Check no overlap with existing mask
        if (total_mask & mask).any():
            raise ValueError("Modality masks overlap - not mutually exclusive")
            
        total_mask |= mask
    
    # Check all positions are covered
    if not total_mask.all():
        raise ValueError("Modality masks don't cover all positions - not collectively exhaustive")
    
    return True

class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.use_mot = args.use_mot
        self.modalities = args.modalities
        self.attention_type = args.attention_type
        
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            if self.use_mot:
                if args.attention_type == "mst":
                    self.layers.append(MSTTransformerBlock(args))
                else:
                    self.layers.append(MoTTransformerBlock(args))
            else:
                self.layers.append(TransformerBlock(args))
    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        modality_ids: Optional[torch.Tensor] = None,
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            if self.use_mot:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl, modality_ids=modality_ids)
            else:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)


class MSTAttention(nn.Module):
    """Multi-Stream Transformer Attention with fine-grained cross-modal and self-modal attention pathways."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        modalities: List[str],
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta
        self.modalities = modalities
        self.heads_per_group = self.n_heads // self.n_kv_heads

        # Modality-specific query projections
        self.wq_m = nn.ModuleDict({
            m: nn.Linear(dim, n_heads * head_dim, bias=False)
            for m in modalities
        })

        # Cross-modal and self-modal key and value projections
        self.wk_m = nn.ModuleDict({
            f"{src}_to_{tgt}": nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            for src in modalities
            for tgt in modalities
        })
        self.wv_m = nn.ModuleDict({
            f"{src}_to_{tgt}": nn.Linear(dim, n_kv_heads * head_dim, bias=False)
            for src in modalities
            for tgt in modalities
        })

        # Modality-specific output projections
        self.wo_m = nn.ModuleDict({
            m: nn.Linear(n_heads * head_dim, dim, bias=False)
            for m in modalities
        })

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        modality_ids: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        N = bsz * seq_len  # Total number of tokens

        # Create modality masks using a mapping from modality IDs to modality names
        modality_ids = modality_ids.view(bsz, seq_len)
        modality_id_to_name = {idx: m for idx, m in enumerate(self.modalities)}
        modality_masks = {
            m: (modality_ids == idx)
            for idx, m in modality_id_to_name.items()
        }

        # If positions are not provided, use default positions
        if tok_idx is not None:
            positions = tok_idx.view(bsz, seq_len)
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)

        # Initialize tensors for Q, K, V
        xq = torch.zeros(N, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        xk = torch.zeros(N, self.n_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
        xv = torch.zeros(N, self.n_kv_heads, self.head_dim, device=x.device, dtype=x.dtype)
        positions_all = torch.zeros(N, device=x.device, dtype=torch.long)

        # Process queries for all target modalities
        for tgt_idx, tgt_mod in modality_id_to_name.items():
            tgt_mask = modality_masks[tgt_mod]
            if not tgt_mask.any():
                continue

            tgt_indices = tgt_mask.nonzero(as_tuple=True)
            flat_indices = tgt_indices[0] * seq_len + tgt_indices[1]
            positions_q = positions[tgt_indices]

            # Get queries for target modality
            q = x[tgt_indices]  # Shape: [N_tgt, dim]
            q = self.wq_m[tgt_mod](q)
            q = q.view(-1, self.n_heads, self.head_dim)

            # Assign to xq
            xq[flat_indices] = q
            positions_all[flat_indices] = positions_q.view(-1)

        # Process keys and values for all source-target modality pairs
        for src_idx, src_mod in modality_id_to_name.items():
            src_mask = modality_masks[src_mod]
            if not src_mask.any():
                continue

            src_indices = src_mask.nonzero(as_tuple=True)
            flat_indices = src_indices[0] * seq_len + src_indices[1]

            src_tokens = x[src_indices]

            # For each target modality, project keys and values
            for tgt_idx, tgt_mod in modality_id_to_name.items():
                kv_proj_key = f"{src_mod}_to_{tgt_mod}"

                k = self.wk_m[kv_proj_key](src_tokens)
                v = self.wv_m[kv_proj_key](src_tokens)

                k = k.view(-1, self.n_kv_heads, self.head_dim)
                v = v.view(-1, self.n_kv_heads, self.head_dim)

                # Accumulate k and v for each flat_index
                xk[flat_indices] += k
                xv[flat_indices] += v

        # Apply rotary embeddings to flattened tensors
        xq, xk = apply_rotary_emb_flattened(xq, xk, freq_cis, positions_all)

        # Reshape back to [bsz, seq_len, ...]
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # Repeat keys and values if necessary
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # Prepare for attention computation
        xq = xq.transpose(1, 2)  # B, n_heads, seq_len, head_dim
        xk = xk.transpose(1, 2)  # B, n_heads, seq_len, head_dim
        xv = xv.transpose(1, 2)  # B, n_heads, seq_len, head_dim

        # Compute attention
        if attn_impl == "sdpa":
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            attn_mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, is_causal=is_causal
            )  # B, n_heads, seq_len, head_dim
            output = output.transpose(1, 2).contiguous()  # B, seq_len, n_heads, head_dim
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported")

        # Reshape output and apply modality-specific output projections
        output = output.view(bsz * seq_len, -1)
        final_output = torch.zeros_like(output)

        # Apply modality-specific output projections
        for tgt_idx, tgt_mod in modality_id_to_name.items():
            tgt_mask = modality_masks[tgt_mod]
            if tgt_mask.any():
                tgt_indices = tgt_mask.nonzero(as_tuple=True)
                flat_indices = tgt_indices[0] * seq_len + tgt_indices[1]
                out_m = output[flat_indices]
                final_m = self.wo_m[tgt_mod](out_m)
                final_output[flat_indices] = final_m

        final_output = final_output.view(bsz, seq_len, -1)

        return final_output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        # Initialize modality-specific projections
        for m in self.modalities:
            nn.init.trunc_normal_(
                self.wq_m[m].weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_(
                self.wo_m[m].weight,
                mean=0.0,
                std=init_std / factor,
                a=-3 * (init_std / factor),
                b=3 * (init_std / factor),
            )
        print(f"MST Attention: Initialized modality-specific projections for {self.modalities} with factor {factor} and std {init_std}")

        # Initialize cross-modal projections
        for key in self.wk_m.keys():
            nn.init.trunc_normal_(
                self.wk_m[key].weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_(
                self.wv_m[key].weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        print(f"MST Attention: Initialized cross-modal projections for pairs: {list(self.wk_m.keys())} with factor {factor} and std {init_std}")

class MSTFeedForward(nn.Module):
    """Multi-Stream Transformer FeedForward with modality-specific layers."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        modalities: List[str],
        mp_size: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.modalities = modalities

        # Modality-specific FFN layers
        self.w1_m = nn.ModuleDict({
            m: nn.Linear(dim, self.hidden_dim, bias=False)
            for m in modalities
        })
        self.w2_m = nn.ModuleDict({
            m: nn.Linear(self.hidden_dim, dim, bias=False)
            for m in modalities
        })
        self.w3_m = nn.ModuleDict({
            m: nn.Linear(dim, self.hidden_dim, bias=False)
            for m in modalities
        })

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        # Create modality masks
        modality_ids = modality_ids.view(bsz, seq_len)
        modality_id_to_name = {idx: m for idx, m in enumerate(self.modalities)}
        modality_masks = {
            m: (modality_ids == idx)
            for idx, m in modality_id_to_name.items()
        }

        output = torch.zeros_like(x)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                x_m = x[indices]
                x1 = self.w1_m[modality](x_m)
                x3 = self.w3_m[modality](x_m)
                out_m = self.w2_m[modality](F.silu(x1) * x3)
                output[indices[0], indices[1]] = out_m
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor

        for m in self.modalities:
            # Initialize w1_m and w3_m
            for w in [self.w1_m[m], self.w3_m[m]]:
                nn.init.trunc_normal_(
                    w.weight,
                    mean=0.0,
                    std=in_init_std,
                    a=-3 * in_init_std,
                    b=3 * in_init_std,
                )
            # Initialize w2_m
            nn.init.trunc_normal_(
                self.w2_m[m].weight,
                mean=0.0,
                std=out_init_std,
                a=-3 * out_init_std,
                b=3 * out_init_std,
            )
        print(f"MST FeedForward: Initialized parameters for modalities: {self.modalities} with factor {factor} and std {init_std}")

class MSTTransformerBlock(nn.Module):
    """Multi-Stream Transformer Block with fine-grained cross-modal attention."""

    def __init__(self, args):
        super().__init__()
        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.dim = args.dim
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // self.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert self.n_heads % self.n_kv_heads == 0
        assert args.dim % self.n_heads == 0

        modalities = args.modalities or ['text', 'image']
        self.modalities = modalities

        # Modality-specific layer norms
        self.attention_norm_m = nn.ModuleDict({
            m: RMSNorm(args.dim, eps=args.norm_eps)
            for m in modalities
        })
        self.ffn_norm_m = nn.ModuleDict({
            m: RMSNorm(args.dim, eps=args.norm_eps)
            for m in modalities
        })

        # MST Attention and FeedForward layers
        self.attention = MSTAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            modalities=modalities,
        )
        self.feed_forward = MSTFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            modalities=modalities,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        modality_ids: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        # Create modality masks using a mapping from modality IDs to modality names
        modality_ids = modality_ids.view(bsz, seq_len)
        modality_id_to_name = {idx: m for idx, m in enumerate(self.modalities)}
        modality_masks = {
            m: (modality_ids == idx)
            for idx, m in modality_id_to_name.items()
        }

        # Apply modality-specific attention layer norms
        x_norm = torch.zeros_like(x)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                x_m = x[indices]
                x_norm_m = self.attention_norm_m[modality](x_m)
                x_norm[indices[0], indices[1]] = x_norm_m

        # Apply attention
        h = x + self.attention(
            x_norm,
            freq_cis,
            modality_ids,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )

        # Apply modality-specific FFN layer norms
        h_norm = torch.zeros_like(h)
        for modality, modality_mask in modality_masks.items():
            if modality_mask.any():
                indices = modality_mask.nonzero(as_tuple=True)
                h_m = h[indices]
                h_norm_m = self.ffn_norm_m[modality](h_m)
                h_norm[indices[0], indices[1]] = h_norm_m

        # Apply feed-forward network
        out = h + self.feed_forward(h_norm, modality_ids)
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        print("MST: Initialized attention parameters")

        for norm in self.attention_norm_m.values():
            norm.reset_parameters()
        print("MST: Initialized attention norm parameters")

        self.feed_forward.reset_parameters(init_std, factor)
        print("MST: Initialized feed-forward parameters")

        for norm in self.ffn_norm_m.values():
            norm.reset_parameters()
        print("MST: Initialized FFN norm parameters")

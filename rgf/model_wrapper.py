"""
Wrapper around a GPT-2-style HuggingFace model that exposes exactly the
objects the RGF paper needs:

  - hidden states h^{(l)}_i(x) at every layer, for a fixed input sequence x
  - the "downstream map" T_{l:L}^{(x,i)}: R^d -> R^{|V|} of Definition 2.5,
    obtained by substituting a hidden state h at (layer l, position i) and
    propagating through the remaining transformer blocks + unembedding,
    holding every other token's representation fixed.

The downstream map is built functionally (no in-place writes) so it can be
passed straight to torch.func.jacfwd / jacrev.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


class RGFModel:
    def __init__(self, pretrained: bool = False, model_name: str = "gpt2", tiny_config: dict | None = None, device: str = "cpu", seed: int = 0):
        """
        pretrained=True  -> downloads real GPT-2 weights from HuggingFace
                             (requires network access to huggingface.co).
        pretrained=False -> builds a randomly-initialized model with the same
                             architecture (GPT2Config), so everything below can
                             be run and unit-tested with no internet access.
        """
        self.device = device
        torch.manual_seed(seed)

        if pretrained:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            cfg_kwargs = dict(
                vocab_size=100,
                n_positions=64,
                n_embd=32,
                n_layer=3,
                n_head=2,
                n_inner=64,
                attn_implementation="eager",
            )
            if tiny_config:
                cfg_kwargs.update(tiny_config)
            config = GPT2Config(**cfg_kwargs)
            self.model = GPT2LMHeadModel(config)
            self.tokenizer = None  # use raw token ids in tests

        self.model.to(device)
        self.model.eval()

        self.L = self.model.config.n_layer         # number of transformer layers
        self.d = self.model.config.n_embd           # hidden dimension
        self.V = self.model.config.vocab_size        # vocab size

    # ------------------------------------------------------------------
    # Forward pass / hidden state extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        input_ids: LongTensor of shape (1, n)
        Returns tuple of length L+1: hidden_states[l] has shape (1, n, d),
        where hidden_states[0] is the embedding output and hidden_states[L]
        is the output of the final transformer block (before ln_f).
        """
        out = self.model.transformer(input_ids, output_hidden_states=True)
        return out.hidden_states  # tuple length L+1

    # ------------------------------------------------------------------
    # Downstream map T_{l:L}^{(x,i)} : R^d -> R^{|V|}   (Definition 2.5)
    # ------------------------------------------------------------------
    def downstream_map(self, hidden_states_l: torch.Tensor, layer_idx: int, position_idx: int):
        """
        hidden_states_l: (1, n, d) tensor -- the *actual* hidden states at
                          layer `layer_idx` for the fixed context x (all
                          positions), as produced by `hidden_states()`.
        layer_idx: l in {0, ..., L}. If l == L, the map reduces to U(h).
        position_idx: i, the token position whose representation is replaced.

        Returns a function f: R^d -> R^{|V|} (batch-free, single vector in,
        single vector out) suitable for torch.func.jacfwd / jacrev.
        """
        hs_fixed = hidden_states_l.detach()
        n = hs_fixed.shape[1]
        blocks = self.model.transformer.h[layer_idx:]  # layers l, l+1, ..., L-1
        ln_f = self.model.transformer.ln_f
        lm_head = self.model.lm_head

        def f(h: torch.Tensor) -> torch.Tensor:
            # h: (d,) -> rebuild the (1, n, d) sequence functionally (no in-place ops)
            h_row = h.reshape(1, 1, -1)
            if position_idx == 0:
                x = torch.cat([h_row, hs_fixed[:, 1:, :]], dim=1)
            elif position_idx == n - 1:
                x = torch.cat([hs_fixed[:, :position_idx, :], h_row], dim=1)
            else:
                x = torch.cat(
                    [hs_fixed[:, :position_idx, :], h_row, hs_fixed[:, position_idx + 1:, :]],
                    dim=1,
                )
            for block in blocks:
                out = block(x)
                x = out[0] if isinstance(out, tuple) else out
            x = ln_f(x)
            logits = lm_head(x)
            return logits[0, position_idx, :]

        return f

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """
        Strictly-affine U(h) = W_U h + b_U for a single vector h, with NO
        final LayerNorm applied. This is the idealized final layer assumed by
        Theorem 6.1. Use this (rather than downstream_map(..., layer_idx=L))
        when you want to check the theorem against the paper's exact
        assumptions; use downstream_map for the real GPT-2 pipeline, which
        inserts ln_f before the unembedding.
        """
        return self.model.lm_head(h.reshape(1, 1, -1))[0, 0, :]

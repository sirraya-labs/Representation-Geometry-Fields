"""
Metric Variation vs. Predictive Uncertainty (v2 — fixed)
==========================================================
Tests the RGF's foundational hypothesis (Proposition 6.1):
Does local metric variation Δ(h) correlate with the model's
own predictive uncertainty (perplexity) about the token h is
about to produce?



References the paper's own notation:
  Δ(h, h') = ||G(h) - G(h')||_F / (0.5 * (||G(h)||_F + ||G(h')||_F))
  Δ(h) = median_{h' in N_k(h)} Δ(h, h')
"""

import torch
import torch.nn as nn
from torch.func import jvp, vjp
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# RGF Metric Variation: Δ(h) via randomized estimation
# ============================================================================

def project(y):
    """Π(y) = y - mean(y) — project onto zero-mean logit space (Section 2)."""
    return y - y.mean()


def estimate_frobenius(downstream_fn, h, num_probes=15):
    """Estimate ||G(h)||_F via Hutchinson randomized probing.

    Uses the identity ||G||_F^2 = E[||G s||^2] for Rademacher s,
    which is unbiased for PSD G (Theorem 4.1).
    """
    d = h.shape[0]
    frob_sq = 0.0
    for _ in range(num_probes):
        s = torch.randint(0, 2, (d,), dtype=h.dtype) * 2 - 1
        s = s / np.sqrt(d)
        _, jvp_out = jvp(downstream_fn, (h,), (s,))
        proj = project(jvp_out)
        _, vjp_fn = vjp(downstream_fn, h)
        Gs, = vjp_fn(proj)
        frob_sq += (Gs ** 2).sum().item()
    return np.sqrt(max(0.0, frob_sq / num_probes))


def compute_delta(downstream_fn, h, n_perturbations=8, n_probes=15):
    """Compute Δ(h) per the paper's definition (Section 7).

    Δ(h) = median_{h'} Δ(h, h') where h' are small perturbations of h.
    Each Δ(h, h') = ||G(h) - G(h')||_F / (0.5*(||G(h)||_F + ||G(h')||_F))
    """
    frob_h = estimate_frobenius(downstream_fn, h, n_probes)
    h_std = h.std().item()
    pert_std = 0.01 * h_std if h_std > 0 else 0.01

    deltas = []
    for _ in range(n_perturbations):
        h_pert = h + torch.randn_like(h) * pert_std
        frob_pert = estimate_frobenius(downstream_fn, h_pert, n_probes)

        diff_sq = 0.0
        half = n_probes // 2
        for _ in range(half):
            s = torch.randint(0, 2, (h.shape[0],), dtype=h.dtype) * 2 - 1
            s = s / np.sqrt(h.shape[0])
            _, jvp_h = jvp(downstream_fn, (h,), (s,))
            _, jvp_p = jvp(downstream_fn, (h_pert,), (s,))
            _, vjp_h = vjp(downstream_fn, h)
            _, vjp_p = vjp(downstream_fn, h_pert)
            Gs_h, = vjp_h(project(jvp_h))
            Gs_p, = vjp_p(project(jvp_p))
            diff_sq += ((Gs_h - Gs_p) ** 2).sum().item()

        diff_frob = np.sqrt(max(0.0, diff_sq / half))
        avg = 0.5 * (frob_h + frob_pert)
        if avg > 1e-10:
            deltas.append(diff_frob / avg)

    return float(np.median(deltas)) if deltas else 0.0


# ============================================================================
# Model Wrapper
# ============================================================================

class GPT2Wrapper:
    def __init__(self, device="cpu"):
        print("Loading GPT-2 (eager attention)...")
        config = GPT2Config.from_pretrained("openai-community/gpt2")
        config.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2", config=config,
            torch_dtype=torch.float32, attn_implementation="eager",
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.layers = self.model.transformer.h
        self.final_norm = self.model.transformer.ln_f
        self.lm_head = self.model.lm_head
        self.num_layers = len(self.layers)
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.device = device
        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_dim}")

    def capture_downstream_context(self, layer_idx, input_ids, attn_mask=None):
        """Run ONE hooked forward pass to capture the kwargs each downstream
        layer (>= layer_idx) needs, plus the cached hidden states. This does
        NOT depend on token position, so it is computed once per
        (sentence, layer) and reused across all positions in that sentence
        (fixes the redundant-recompute issue in v1).
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask,
                                  output_hidden_states=True)
            cached = outputs.hidden_states[layer_idx].clone()

        captured = {}
        def hook_factory(idx):
            def hook(module, args, kwargs):
                captured[idx] = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                                  for k, v in kwargs.items()}
            return hook

        hooks = [self.layers[l].register_forward_pre_hook(hook_factory(l), with_kwargs=True)
                 for l in range(layer_idx, self.num_layers)]
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attn_mask)
        for h in hooks:
            h.remove()

        return cached, captured, outputs

    def make_downstream_fn(self, layer_idx, pos, cached, captured):
        """Build the downstream map for a specific position, reusing a
        (sentence, layer)-level context captured once via
        capture_downstream_context.
        """
        def fn(h):
            hidden = cached.clone()
            hidden[0, pos, :] = h
            for l in range(layer_idx, self.num_layers):
                kw = dict(captured.get(l, {}))
                out = self.layers[l](hidden, **kw)
                hidden = out[0] if isinstance(out, tuple) else out
            return self.lm_head(self.final_norm(hidden))[0, pos, :]
        return fn


# ============================================================================
# Experiment: Δ(h) vs. Perplexity
# ============================================================================

SENTENCES = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The Earth orbits around the Sun.",
    "William Shakespeare wrote Hamlet and Macbeth.",
    "The United Nations was founded in 1945.",
    "Photosynthesis converts sunlight into energy.",
    "The Great Wall of China is visible from space.",
    "Isaac Newton discovered the laws of gravity.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Beethoven composed the Ninth Symphony.",
    "The printing press was invented by Gutenberg.",
    "Diamonds are formed under extreme pressure.",
    "The Amazon rainforest produces 20% of Earth's oxygen.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "The speed of light is approximately 300,000 km/s.",
    "Mount Everest is the highest mountain on Earth.",
    "The human body has 206 bones.",
    "Vincent van Gogh painted Starry Night.",
    "The Industrial Revolution began in Britain.",
    "DNA carries genetic information in all living organisms.",
    "The Renaissance was a period of cultural rebirth.",
    "Oxygen is essential for human respiration.",
    "The Roman Empire fell in 476 AD.",
    "Marie Curie discovered radium and polonium.",
    "The first moon landing occurred in 1969.",
    "The purple elephant danced on a rainbow cloud.",
    "My toaster wrote a symphony for kitchen utensils.",
    "The invisible giraffe painted the sky purple yesterday.",
    "A quantum potato decided to learn ballet dancing.",
    "The sentient lampshade debated philosophy with a sock.",
    "My shadow learned to play the accordion last Tuesday.",
    "The gravity-defying sandwich escaped from the fridge.",
    "A time-traveling teaspoon invented the color blue.",
    "The singing mailbox delivered emotions instead of letters.",
    "My left shoe became prime minister of the moon.",
    "The philosophical brick contemplated its existence.",
]


def compute_perplexity(logits, target_ids):
    """Per-token perplexity from logits. perplexity = exp(cross_entropy)."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    nll = -log_probs[torch.arange(len(target_ids)), target_ids]
    return torch.exp(nll)


def run_experiment(layers_to_test=(4, 7, 10), output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = GPT2Wrapper(device=device)

    all_results = []
    n_failed = 0

    for layer_idx in layers_to_test:
        print(f"\n{'='*50}")
        print(f"Layer {layer_idx}")
        print(f"{'='*50}")

        for sent_id, text in enumerate(tqdm(SENTENCES, desc=f"Layer {layer_idx}")):
            inputs = wrapper.tokenizer(text, return_tensors='pt').to(device)
            seq_len = inputs['input_ids'].shape[1]
            if seq_len < 2:
                continue

            # Capture hook context ONCE per (sentence, layer) — fix #2.
            try:
                cached, captured, outputs = wrapper.capture_downstream_context(
                    layer_idx, inputs['input_ids'], inputs.get('attention_mask'))
            except Exception:
                n_failed += seq_len
                continue

            logits_full = outputs.logits[0]  # [seq_len, vocab]

            # Positions 0 .. seq_len-2: each h at `pos` predicts token at
            # pos+1 (fix #1 — aligned target). The last position has no
            # next token, so it is excluded.
            for pos in range(0, seq_len - 1):
                try:
                    fn = wrapper.make_downstream_fn(layer_idx, pos, cached, captured)
                    h = outputs.hidden_states[layer_idx][0, pos, :].clone()

                    delta = compute_delta(fn, h, n_perturbations=5, n_probes=12)

                    target_id = inputs['input_ids'][0, pos + 1].item()
                    next_logits = logits_full[pos:pos + 1, :]
                    ppl = compute_perplexity(next_logits, torch.tensor([target_id])).item()

                    all_results.append({
                        'layer': layer_idx,
                        'sentence_id': sent_id,
                        'position': pos,
                        'delta': delta,
                        'perplexity': ppl,
                        'text': text[:50],
                    })
                except Exception:
                    n_failed += 1
                    continue

    print(f"\nTotal failed (delta, perplexity) computations: {n_failed}")

    # ====================================================================
    # Analysis per layer: pooled (per-token) + sentence-level robustness
    # ====================================================================
    print(f"\n{'='*60}")
    print("Results: Δ(h) vs. Perplexity Correlation")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, len(layers_to_test), figsize=(6 * len(layers_to_test), 5))
    if len(layers_to_test) == 1:
        axes = [axes]

    summary_rows = []

    for i, layer_idx in enumerate(layers_to_test):
        layer_data = [r for r in all_results if r['layer'] == layer_idx]
        deltas = np.array([r['delta'] for r in layer_data])
        ppls = np.array([r['perplexity'] for r in layer_data])

        rho, p = stats.spearmanr(deltas, ppls)
        r, p_pearson = stats.pearsonr(deltas, ppls)

        # Sentence-level robustness check: average delta/perplexity per
        # sentence, then correlate across sentences (fix #3). This avoids
        # treating within-sentence tokens as independent samples.
        sent_ids = sorted(set(r_['sentence_id'] for r_ in layer_data))
        sent_deltas, sent_ppls = [], []
        for sid in sent_ids:
            sd = [r_['delta'] for r_ in layer_data if r_['sentence_id'] == sid]
            sp = [r_['perplexity'] for r_ in layer_data if r_['sentence_id'] == sid]
            if sd:
                sent_deltas.append(np.mean(sd))
                sent_ppls.append(np.mean(sp))
        sent_deltas = np.array(sent_deltas)
        sent_ppls = np.array(sent_ppls)
        if len(sent_deltas) > 2:
            rho_sent, p_sent = stats.spearmanr(sent_deltas, sent_ppls)
        else:
            rho_sent, p_sent = float('nan'), float('nan')

        print(f"\nLayer {layer_idx} ({len(layer_data)} tokens, {len(sent_ids)} sentences):")
        print(f"  Δ range: [{deltas.min():.4f}, {deltas.max():.4f}]")
        print(f"  PPL range: [{ppls.min():.1f}, {ppls.max():.1f}]")
        print(f"  Pooled (per-token):   Spearman ρ = {rho:.3f}, p = {p:.4f}  "
              f"({'✓' if p < 0.05 else '✗'})")
        print(f"  Pearson r = {r:.3f}, p = {p_pearson:.4f}")
        print(f"  Sentence-level (mean per sentence): Spearman ρ = {rho_sent:.3f}, "
              f"p = {p_sent:.4f}  ({'✓' if p_sent < 0.05 else '✗'})")

        summary_rows.append({
            'layer': layer_idx, 'n_tokens': len(layer_data), 'n_sentences': len(sent_ids),
            'rho_pooled': rho, 'p_pooled': p,
            'rho_sentence': rho_sent, 'p_sentence': p_sent,
        })

        ax = axes[i]
        ax.scatter(ppls, deltas, alpha=0.4, s=15, color='#2196F3', label='per-token')
        ax.scatter(sent_ppls, sent_deltas, alpha=0.9, s=40, color='#E65100',
                   marker='D', label='per-sentence mean')
        if len(deltas) > 2:
            z = np.polyfit(ppls, deltas, 1)
            x_line = np.linspace(ppls.min(), ppls.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Per-Token Perplexity (next-token)')
        ax.set_ylabel('Local Metric Variation Δ(h)')
        ax.set_title(f'Layer {layer_idx}\nρ_pooled={rho:.3f} (p={p:.3f}), '
                      f'ρ_sent={rho_sent:.3f} (p={p_sent:.3f})')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('RGF Metric Variation vs. Predictive Uncertainty (v2, aligned)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/delta_vs_perplexity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved: {output_dir}/delta_vs_perplexity.png")

    # ====================================================================
    # Summary table for paper
    # ====================================================================
    print(f"\n{'='*70}")
    print("Paper-Ready Summary")
    print(f"{'='*70}")
    print(f"{'Layer':<7}{'N tok':<8}{'N sent':<8}{'ρ_pooled':<11}{'p_pooled':<11}"
          f"{'ρ_sent':<10}{'p_sent':<10}")
    print(f"{'-'*65}")
    for row in summary_rows:
        print(f"{row['layer']:<7}{row['n_tokens']:<8}{row['n_sentences']:<8}"
              f"{row['rho_pooled']:<11.3f}{row['p_pooled']:<11.4f}"
              f"{row['rho_sentence']:<10.3f}{row['p_sentence']:<10.4f}")

    with open(f"{output_dir}/delta_perplexity_results.json", 'w') as f:
        json.dump({'per_token_results': all_results, 'summary': summary_rows,
                    'n_failed': n_failed}, f, indent=2)

    return all_results, summary_rows


def main():
    run_experiment(layers_to_test=(4, 7, 10), output_dir="outputs")


if __name__ == "__main__":
    main()
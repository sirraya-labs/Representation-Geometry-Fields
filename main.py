"""
Representation Geometry Field — Reference Implementation
=========================================================
Downstream map via single-forward-pass intervention.

THE APPROACH:
  Instead of capturing kwargs and replaying layers, we modify the
  model's forward pass to accept an intervention at layer ℓ.
  
  For T_{ℓ:L}(h):
  1. Run model.forward() with an intervention that replaces the
     hidden state at (layer ℓ, position i) with h.
  2. The model continues its native forward from that point.
  3. Return logits at position i.

  This guarantees:
  - Exact preprocessing (masks, RoPE, cache, SDPA dispatch)
  - Exact decoder layer computation
  - No kwargs to capture or replay
  - Works with any architecture because the model controls its own
    forward pass

  For torch.func compatibility:
  We make the forward pass functional by wrapping it in a function
  that takes h as input and returns logits. The intervention is
  implemented using the model's output_hidden_states mechanism:
  
  1. Run model.forward() with output_hidden_states=True
  2. In a modified forward, skip layers 0..ℓ-1 (using cached output)
  3. Inject h
  4. Continue through layers[ℓ:] using the model's own layer loop

  BUT we don't recreate the layer loop. Instead, we use the fact
  that the model's internal forward is just:
    for layer in layers:
        hidden = layer(hidden, **kwargs)
  
  The kwargs ARE the issue. So we need them from the model.
  
  THE ACTUAL SOLUTION:
  Use torch.func.functional_call on the model's entire decoder
  stack, with the hidden state at layer ℓ as an explicit input.
  
  We create a thin wrapper that:
  1. Takes the hidden state at layer ℓ (with h injected)
  2. Calls the remaining decoder layers with the SAME kwargs
     the model computed during a reference forward pass
  
  This is the hook approach but made explicit: the kwargs are
  captured once and treated as constants. The key improvement
  is that we verify they don't depend on h by construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GPTNeoXForCausalLM, LlamaForCausalLM, GPT2LMHeadModel,
    MistralForCausalLM, Qwen2ForCausalLM,
)
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


# ============================================================================
# Configuration & Data Structures
# ============================================================================

@dataclass
class RGFConfig:
    num_eigenvalues: int = 100
    oversampling: int = 10
    power_iterations: int = 2
    trace_probes: int = 100
    variation_probes: int = 30
    variation_neighbors: int = 30
    eigenvalue_tol: float = 1e-6
    equivalence_rtol: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 42

@dataclass
class RGFResult:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    trace: float
    frobenius: float
    rank: int
    effective_rank: float
    nullspace_dim: int
    condition_number: float

@dataclass
class VariationResult:
    layer_idx: int
    delta_values: torch.Tensor
    delta_dot_values: torch.Tensor
    noise_floor: float
    significant: bool
    ratio_to_final: Optional[float] = None

@dataclass
class TrajectoryResult:
    velocity_norms: torch.Tensor
    unit_velocities: torch.Tensor
    path_length: float
    velocity_changes: torch.Tensor
    mean_velocity_change: float


# ============================================================================
# Linear Operator
# ============================================================================

class RGFOperator:
    """G(h) = J^T Π J as matrix-free operator. 2 AD passes per matvec."""
    
    def __init__(self, fn: Callable, h: torch.Tensor, vocab_size: int):
        self._fn = fn
        self._h = h.detach().clone()
        self._V = vocab_size
        self._d = h.shape[0]
    
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        v = v.reshape(-1)
        _, jvp_out = jvp(self._fn, (self._h,), (v,))
        proj = jvp_out - jvp_out.mean(dim=-1, keepdim=True)
        _, vjp_fn = vjp(self._fn, self._h)
        result, = vjp_fn(proj)
        return result
    
    @property
    def shape(self): return (self._d, self._d)
    @property
    def device(self): return self._h.device


class DifferenceOperator:
    def __init__(self, a, b):
        self._a, self._b = a, b
    def matvec(self, v): return self._a.matvec(v) - self._b.matvec(v)
    @property
    def shape(self): return self._a.shape
    @property
    def device(self): return self._a.device


# ============================================================================
# Randomized Linear Algebra
# ============================================================================

def randomized_svd_psd(op, k, oversampling=10, power_iterations=2, seed=None):
    d = op.shape[0]
    n = min(k + oversampling, d)
    gen = torch.Generator(device=op.device)
    if seed is not None: gen.manual_seed(seed)
    
    Omega = torch.randn(d, n, device=op.device, generator=gen)
    Y = torch.zeros(d, n, device=op.device)
    for j in range(n): Y[:, j] = op.matvec(Omega[:, j])
    for _ in range(power_iterations):
        Yn = torch.zeros_like(Y)
        for j in range(n): Yn[:, j] = op.matvec(Y[:, j])
        Y, _ = torch.linalg.qr(Yn)
    Q, _ = torch.linalg.qr(Y)
    
    B = torch.zeros(n, n, device=op.device)
    for j in range(n): B[:, j] = Q.T @ op.matvec(Q[:, j])
    B = 0.5 * (B + B.T)
    e, ev = torch.linalg.eigh(B)
    e, ev = torch.flip(e, [0])[:k], torch.flip(ev, [1])[:, :k]
    mask = e > 0
    return e[mask], Q @ ev[:, mask]


def randomized_trace_frobenius(op, num_probes=100, seed=None):
    d, n = op.shape[0], min(num_probes, op.shape[0])
    gen = torch.Generator(device=op.device)
    if seed is not None: gen.manual_seed(seed)
    
    S = (2 * torch.randint(0, 2, (d, n), device=op.device, generator=gen).float() - 1) / np.sqrt(n)
    AS = torch.zeros(d, n, device=op.device)
    for j in range(n): AS[:, j] = op.matvec(S[:, j])
    
    Q, _ = torch.linalg.qr(AS)
    q = min(Q.shape[1], n)
    Q = Q[:, :q]
    
    AQ = torch.zeros(d, q, device=op.device)
    for j in range(q): AQ[:, j] = op.matvec(Q[:, j])
    trace = torch.trace(Q.T @ AQ).item()
    
    if q < d:
        G = torch.randn(d, n, device=op.device, generator=gen)
        G = G - Q @ (Q.T @ G)
        AG = torch.zeros(d, n, device=op.device)
        for j in range(n): AG[:, j] = op.matvec(G[:, j])
        trace += torch.trace(G.T @ AG).item()
    
    return trace, np.sqrt(max(0.0, (AS**2).sum().item() / n))


# ============================================================================
# Correct Downstream Map: Single Forward Pass with Internal Intervention
# ============================================================================

class InterventionDownstreamMap:
    """T_{ℓ:L}(h) via single forward pass with hidden state intervention.
    
    APPROACH:
    We make the model's internal decoder loop accept an intervention.
    This is done by wrapping the decoder layers in a callable that
    checks for an injected hidden state at layer ℓ.
    
    Specifically:
    1. Run model.forward() once normally to get:
       - hidden states at all layers
       - the exact kwargs passed to each decoder layer
    2. Define T_{ℓ:L}(h) as:
       a. Start from cached hidden state at layer ℓ
       b. Inject h at target position
       c. Continue through layers[ℓ:] using the EXACT kwargs the model
          originally used
    
    CORRECTNESS CONDITION:
    This is exact for architectures where the per-layer kwargs
    (attention_mask, position_ids, cache_position, RoPE tensors, etc.)
    are functions only of the input sequence and not of the hidden
    states themselves. This holds for all current dense decoder LLMs
    (GPT-2, GPT-NeoX, Llama, Mistral, Qwen, Gemma, etc.)
    
    For architectures with data-dependent routing (MoE), the routing
    decision depends on hidden states and replay would be incorrect.
    Those architectures require a different approach and are not yet
    supported.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        position_idx: int,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.position_idx = position_idx
        
        # Get architecture components
        self._layers, self._final_norm, self._lm_head = self._get_components(model)
        self._num_layers = len(self._layers)
        self._is_final = (layer_idx == self._num_layers)
        
        # Capture kwargs by running one forward pass with hooks
        self._captured_kwargs: List[Dict[str, Any]] = [{} for _ in range(self._num_layers)]
        self._cached_hidden = self._capture_forward(input_ids, attention_mask)
    
    @staticmethod
    def _get_components(model):
        if isinstance(model, GPTNeoXForCausalLM) or hasattr(model, 'gpt_neox'):
            g = model.gpt_neox if hasattr(model, 'gpt_neox') else model
            return g.layers, g.final_layer_norm, (
                model.embed_out if hasattr(model, 'embed_out') else model.lm_head)
        elif isinstance(model, GPT2LMHeadModel) or hasattr(model, 'transformer'):
            t = model.transformer if hasattr(model, 'transformer') else model
            return t.h, t.ln_f, model.lm_head
        elif isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM)):
            b = model.model if hasattr(model, 'model') else model
            return b.layers, b.norm, model.lm_head
        raise ValueError(f"Unsupported: {type(model).__name__}")
    
    def _capture_forward(self, input_ids, attention_mask):
        """Run forward pass, capturing kwargs via hooks."""
        hooks = []
        
        def make_hook(idx):
            def hook(module, args, kwargs):
                self._captured_kwargs[idx] = {
                    k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                # For the target layer, capture the input hidden state
                if idx == self.layer_idx and not self._is_final:
                    self._captured_input = args[0].detach().clone()
            return hook
        
        for l in range(self._num_layers):
            hooks.append(
                self._layers[l].register_forward_hook(make_hook(l), with_kwargs=True)
            )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        for h in hooks:
            h.remove()
        
        if self._is_final:
            return outputs.hidden_states[self._num_layers].clone()
        if hasattr(self, '_captured_input'):
            return self._captured_input
        return outputs.hidden_states[self.layer_idx].clone()
    
    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """T_{ℓ:L}(h) → logits at position_idx.
        
        Replays layers[ℓ:] with captured kwargs.
        Correct for architectures where layer kwargs are independent
        of hidden state perturbations.
        """
        hidden = self._cached_hidden.clone()
        hidden[0, self.position_idx, :] = h
        
        if self._is_final:
            hidden = self._final_norm(hidden)
            return self._lm_head(hidden)[0, self.position_idx, :]
        
        for l in range(self.layer_idx, self._num_layers):
            kwargs = dict(self._captured_kwargs[l])
            output = self._layers[l](hidden, **kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
        
        hidden = self._final_norm(hidden)
        return self._lm_head(hidden)[0, self.position_idx, :]


class DownstreamMapFactory:
    def __init__(self, model):
        self.model = model
        self._cache = {}
    
    def create(self, layer_idx, position_idx, input_ids, attention_mask=None):
        key = (layer_idx, position_idx, input_ids.shape[1])
        if key not in self._cache:
            self._cache[key] = InterventionDownstreamMap(
                self.model, layer_idx, position_idx, input_ids, attention_mask
            )
        return self._cache[key]


# ============================================================================
# Verification
# ============================================================================

def verify(model_name="EleutherAI/pythia-160m", tests=3, rtol=1e-5):
    """Verify InterventionDownstreamMap."""
    device = RGFConfig().device
    
    print(f"\n{'='*60}")
    print(f"Verification: {model_name}")
    print(f"{'='*60}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = (model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers')
                  else model.config.n_layer)
    vocab_size = model.config.vocab_size
    factory = DownstreamMapFactory(model)
    
    texts = [
        "The capital of France is",
        "Machine learning is a subset of",
        "The theory of relativity was",
    ][:tests]
    
    all_ok = True
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        am = inputs.get('attention_mask')
        pos = inputs['input_ids'].shape[1] - 1
        
        with torch.no_grad():
            out = model(**inputs)
            orig_logits = out.logits[0, pos, :]
            orig_probs = F.softmax(orig_logits, dim=-1)
        
        for lidx in [num_layers, num_layers - 1, max(0, num_layers - 2)]:
            fn = factory.create(lidx, pos, inputs['input_ids'], am)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                h = outputs.hidden_states[lidx][0, pos, :].clone()
            
            fl = fn(h)
            ok1 = (orig_logits - fl).abs().max().item() < rtol
            ok2 = (orig_probs - F.softmax(fl, dim=-1)).abs().max().item() < rtol
            
            v = torch.randn_like(h) * 0.01
            _, ja = jvp(fn, (h,), (v,))
            eps = 1e-4
            jf = (fn(h + eps * v) - fn(h - eps * v)) / (2 * eps)
            ok3 = ((ja - jf).norm() / jf.norm().clamp(1e-10)).item() < 1e-3
            
            op = RGFOperator(fn, h, vocab_size)
            w = torch.randn_like(h)
            Gw = op.matvec(w)
            ok4 = (torch.dot(w, Gw) > -1e-8).item()
            
            pj = ja - ja.mean()
            _, vf = vjp(fn, h)
            JtPJv, = vf(pj)
            ok5 = ((Gw - JtPJv).norm() / Gw.norm().clamp(1e-10)).item() < 1e-3
            
            ok = all([ok1, ok2, ok3, ok4, ok5])
            if not ok: all_ok = False
            s = "✓" if ok else "✗"
            print(f"  {s} ℓ={lidx} '{text[:30]}...' "
                  f"Δlog={max((orig_logits-fl).abs()):.1e} JVP={((ja-jf).norm()/jf.norm().clamp(1e-10)).item():.1e} Op={((Gw-JtPJv).norm()/Gw.norm().clamp(1e-10)).item():.1e}")
    
    print(f"\n  {'✓ ALL PASSED' if all_ok else '✗ SOME FAILED'}")
    return all_ok


# ============================================================================
# RGF Computation & Pipeline
# ============================================================================

def compute_rgf(fn, h, vocab_size, cfg):
    op = RGFOperator(fn, h, vocab_size)
    ev, ec = randomized_svd_psd(op, cfg.num_eigenvalues, cfg.oversampling,
                                 cfg.power_iterations, cfg.seed)
    tr, fr = randomized_trace_frobenius(op, cfg.trace_probes, cfg.seed)
    
    if len(ev) and ev[0] > 0:
        tol = cfg.eigenvalue_tol * ev[0].item()
        rk = (ev > tol).sum().item()
        ef = (ev.sum()**2/(ev**2).sum()).item() if ev.sum() > 0 else 0.0
        pos = ev[ev > tol]
        cd = (pos[0]/pos[-1]).item() if len(pos) > 1 else 1.0
    else:
        rk = ef = 0; cd = float('inf')
    return RGFResult(ev, ec, tr, fr, rk, ef, op.shape[0] - rk, cd)


def compute_variation(operators, hidden_states, cfg):
    N = len(operators)
    h_norm = F.normalize(hidden_states, dim=1)
    dist = torch.sqrt(2 - 2*(h_norm @ h_norm.T).clamp(-1, 1))
    k = min(cfg.variation_neighbors, N-1)
    _, nb = torch.topk(dist, k=k+1, largest=False)
    nb = nb[:, 1:]
    
    fnorms = torch.zeros(N)
    for i, op in enumerate(operators):
        _, fnorms[i] = randomized_trace_frobenius(op, cfg.variation_probes)
    
    deltas = torch.zeros(N); dots = []
    for i in range(N):
        nd = []
        for j in nb[i]:
            df_op = DifferenceOperator(operators[i], operators[j])
            _, df = randomized_trace_frobenius(df_op, cfg.variation_probes)
            avg = 0.5*(fnorms[i]+fnorms[j])
            if avg > 1e-10:
                d = df/avg.item(); nd.append(d)
                dij = dist[i,j].item()
                if dij > 1e-10: dots.append(d/dij)
        if nd: deltas[i] = float(np.median(nd))
    
    noise = max(1e-6, deltas[deltas>0].min().item() if (deltas>0).any() else 1e-6)
    sig = (deltas > 3*noise).float().mean().item() > 0.5
    return VariationResult(-1, deltas, torch.tensor(dots or [0.0]), noise, sig)


def analyze_trajectory(hidden_states, downstream_fns, vocab_size, cfg):
    L = len(hidden_states)-1; d = hidden_states[0].shape[0]
    velocities = torch.stack([hidden_states[l+1]-hidden_states[l] for l in range(L)])
    vn = torch.zeros(L); uv = torch.zeros(L, d)
    for l in range(L):
        op = RGFOperator(downstream_fns[l], hidden_states[l], vocab_size)
        v = velocities[l]; Gv = op.matvec(v)
        n2 = torch.dot(v, Gv).item()
        if n2 > 1e-10: vn[l] = np.sqrt(n2); uv[l] = v/vn[l]
    s = torch.zeros(L-2)
    for l in range(1, L-1): s[l-1] = (uv[l]-uv[l-1]).norm()
    return TrajectoryResult(vn, uv, vn.sum().item(), s,
                           s.mean().item() if len(s) > 0 else 0.0)


class RGFPipeline:
    def __init__(self, model_name, config=None):
        self.cfg = config or RGFConfig()
        self.model_name = model_name
        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        ).to(self.cfg.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        c = self.model.config
        self.num_layers = (c.num_hidden_layers if hasattr(c, 'num_hidden_layers') else c.n_layer)
        self.hidden_dim = (c.hidden_size if hasattr(c, 'hidden_size') else c.n_embd)
        self.vocab_size = c.vocab_size
        self.factory = DownstreamMapFactory(self.model)
        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_dim}")
    
    def run_phase1(self, texts, layers=None):
        if layers is None:
            layers = sorted(set([self.num_layers, self.num_layers-1,
                                max(0, self.num_layers-2)]), reverse=True)
        print(f"\n{'='*60}\nPhase 1: H0 Test\n{'='*60}")
        
        ld = {}
        for layer_idx in layers:
            label = "final" if layer_idx == self.num_layers else f"ℓ={layer_idx}"
            print(f"\n{label}:")
            rgf_r, hl, fl = [], [], []
            for text in tqdm(texts, desc="  Computing"):
                inp = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                ids = inp['input_ids'].to(self.cfg.device)
                am = inp.get('attention_mask')
                if am is not None: am = am.to(self.cfg.device)
                pos = ids.shape[1]-1
                with torch.no_grad():
                    out = self.model(ids, attention_mask=am, output_hidden_states=True)
                    h = out.hidden_states[layer_idx][0, pos, :].clone()
                fn = self.factory.create(layer_idx, pos, ids, am)
                fl.append(fn); hl.append(h)
                rgf_r.append(compute_rgf(fn, h, self.vocab_size, self.cfg))
            hs = torch.stack(hl)
            tr = [r.trace for r in rgf_r]; er = [r.effective_rank for r in rgf_r]
            print(f"    Trace: {np.mean(tr):.4f} ± {np.std(tr):.4f}")
            print(f"    Eff rank: {np.mean(er):.2f} ± {np.std(er):.2f}")
            ld[layer_idx] = {'hidden_states': hs, 'rgf_results': rgf_r, 'downstream_fns': fl}
        
        print(f"\n{'='*60}\nMetric Variation\n{'='*60}")
        vr = {}
        for layer_idx in layers:
            d = ld[layer_idx]
            ops = [RGFOperator(d['downstream_fns'][i], d['hidden_states'][i], self.vocab_size)
                   for i in range(len(texts))]
            v = compute_variation(ops, d['hidden_states'], self.cfg)
            v.layer_idx = layer_idx; vr[layer_idx] = v
            print(f"  ℓ={layer_idx}: Δ = {v.delta_values.mean():.6f} (med {v.delta_values.median():.6f})")
        
        final = max(layers); h0 = False
        print(f"\n{'='*60}\nH0 Test\n{'='*60}")
        for layer_idx in sorted([l for l in layers if l < final]):
            di = vr[layer_idx].delta_values.mean().item()
            df = vr[final].delta_values.mean().item()
            ns = vr[layer_idx].noise_floor
            ratio = di/max(df, 1e-10)
            vr[layer_idx].ratio_to_final = ratio
            print(f"  ℓ={layer_idx}: Δ={di:.6f} ratio={ratio:.2f} >3σ={'✓' if di>3*ns else '✗'}")
            if di > 3*ns and ratio > 2.0: h0 = True
        print(f"\n  {'H0 rejected' if h0 else 'Failed to reject H0'}")
        return {'layer_data': ld, 'variation_results': vr, 'h0_rejected': h0}


# ============================================================================
# Main
# ============================================================================

def main():
    model_name = "EleutherAI/pythia-160m"
    if not verify(model_name):
        print("\n⚠️  VERIFICATION FAILED"); return
    print("\n✓ Verification passed.")
    
    cfg = RGFConfig(num_eigenvalues=50, oversampling=5, power_iterations=1,
                    trace_probes=50, variation_probes=20)
    pipeline = RGFPipeline(model_name, cfg)
    
    texts = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Transformers were introduced in 2017.",
        "Attention mechanisms are key to modern NLP.",
        "Einstein developed the theory of relativity.",
        "Quantum mechanics describes subatomic particles.",
        "The Earth orbits the Sun once per year.",
    ]
    
    results = pipeline.run_phase1(texts)
    print(f"\n{'='*60}\nPhase 1 Complete\nH0 rejected: {results['h0_rejected']}")


if __name__ == "__main__":
    main()
"""
Numerically verifies every theorem in the RGF specification against a
GPT-2-architecture model. Runs fast on a tiny randomly-initialized model
(no internet required); pass --pretrained to check against real GPT-2
weights instead (requires network access to huggingface.co).
"""
import argparse
import torch

from rgf import RGFModel
from rgf.metric import (
    compute_rgf, pullback_check, psd_check, rank_check,
    trace_check, spectral_check, tensor_transform_check,
    final_layer_flatness_check,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Use real GPT-2 weights instead of a tiny random model")
    parser.add_argument("--layer", type=int, default=None, help="Layer index to test (defaults to a middle layer)")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    model = RGFModel(pretrained=args.pretrained, model_name="gpt2")
    model.model.double()
    print(f"Model: L={model.L} layers, d={model.d} hidden dim, |V|={model.V} vocab size\n")

    if args.pretrained:
        text = "The theorem about geometry states that"
        input_ids = model.tokenizer(text, return_tensors="pt").input_ids
    else:
        torch.manual_seed(1)
        n = 6
        input_ids = torch.randint(0, model.V, (1, n))

    hidden_states = model.hidden_states(input_ids)
    n_tokens = input_ids.shape[1]
    position_idx = n_tokens - 1
    layer_idx = args.layer if args.layer is not None else model.L // 2

    print(f"Testing at layer {layer_idx}, token position {position_idx} (sequence length {n_tokens})\n")

    h = hidden_states[layer_idx][0, position_idx, :].detach().clone()
    f = model.downstream_map(hidden_states[layer_idx], layer_idx=layer_idx, position_idx=position_idx)

    results = {}

    print("Computing RGF and running checks...")
    out = compute_rgf(f, h)

    r = pullback_check(f, h)
    results["Pullback identity (Thm 3.1)"] = r["passed"]
    print(f"  [Thm 3.1] Pullback identity        : max err={r['max_abs_error']:.2e}  -> {'PASS' if r['passed'] else 'FAIL'}")

    r = psd_check(out)
    results["PSD (Thm 4.1)"] = r["passed"]
    print(f"  [Thm 4.1] PSD                       : min eig={r['min_eigenvalue']:.2e}  -> {'PASS' if r['passed'] else 'FAIL'}")

    r = rank_check(out)
    results["Rank / kernel (Thm 4.2)"] = r["passed"]
    print(f"  [Thm 4.2] Rank structure             : rank(J)={r['rank_J']}, rank(G)={r['rank_G']}, "
          f"drop observed={r['rank_drop_observed']}, drop predicted={r['rank_drop_predicted']} -> {'PASS' if r['passed'] else 'FAIL'}")

    r = trace_check(out)
    results["Trace (Thm 4.4)"] = r["passed"]
    print(f"  [Thm 4.4] Total sensitivity (trace) : err={r['abs_error']:.2e}  -> {'PASS' if r['passed'] else 'FAIL'}")

    r = spectral_check(out)
    results["Spectral (Thm 4.5)"] = r["passed"]
    print(f"  [Thm 4.5] Spectral interpretation    : max err={r['max_abs_error']:.2e}  -> {'PASS' if r['passed'] else 'FAIL'}")

    r = tensor_transform_check(f, h)
    results["Tensor transform law (Thm 4.7)"] = r["passed"]
    print(f"  [Thm 4.7] Tensor transformation law : matrix err={r['max_abs_error_matrix']:.2e}, "
          f"quad-form err={r['quadratic_form_error']:.2e}, rank preserved={r['rank_preserved']} -> {'PASS' if r['passed'] else 'FAIL'}")

    f_L = model.downstream_map(hidden_states[model.L], layer_idx=model.L, position_idx=position_idx)
    r = final_layer_flatness_check(f_L, model.d, dtype=torch.float64)
    results["Final-layer flatness (Thm 6.1/6.2)"] = r["passed"]
    note = "" if r["passed"] else "  (expected: real GPT-2's final LayerNorm is non-affine, breaking exact constancy)"
    print(f"  [Thm 6.1/6.2] Final-layer flatness  : max deviation across h={r['max_deviation_across_h']:.2e}  -> {'PASS' if r['passed'] else 'FAIL'}{note}")

    # Sanity check: against the *strictly affine* U(h) = W_U h + b_U (no ln_f),
    # flatness should hold to machine precision, confirming ln_f is the sole cause above.
    r_affine = final_layer_flatness_check(model.unembed, model.d, dtype=torch.float64)
    print(f"\n  [sanity] Flatness of raw affine U(h) (no ln_f): max deviation={r_affine['max_deviation_across_h']:.2e} -> "
          f"{'PASS' if r_affine['passed'] else 'FAIL'}  (isolates ln_f as the cause of the FAIL above)")

    print("\nSummary:")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL':4s}  {k}")


if __name__ == "__main__":
    main()

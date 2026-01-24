#!/usr/bin/env python3
"""
simulate_truthcert_vs_rct_pack.py

This does NOT run an LLM. It's a *stress model* conditioned on a single real paper,
to answer a narrow question: does adding TC-RCT validation always add value?

Answer: no — it reduces false-ships for certain RCT-specific corruptions, but it
can increase false rejects on clean extractions (and costs extra tokens).

Usage:
  python simulate_truthcert_vs_rct_pack.py --n 5000 --seed 2026
"""
import argparse, random, math
import pandas as pd

# Import the toy TruthCert protocol
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "truthcert_toy_benchmark.py"
spec = importlib.util.spec_from_file_location("truthcert_toy_benchmark", BASE)
tc = importlib.util.module_from_spec(spec)
sys.modules["truthcert_toy_benchmark"] = tc
spec.loader.exec_module(tc)

# A "paper profile": realistic-ish parameters for a clean, single-table RCT PDF
PAPER_PROFILE = dict(
    n_fields=90,
    n_critical_fields=16,
    corruption_rate=0.18,          # some extraction noise / occasional PDF weirdness
    parser_instability_rate=0.12,  # occasional layout parsing issues
    mixing_pressure=0.08,          # low (single paper) but not zero (source mixing)
    uncertainty_rate=0.10,         # modest (interpretation + rounding)
)

# TC-RCT pack (toy) — parameters chosen to represent a strong-but-not-perfect validator
PACK = dict(
    token_cost=260,
    # True-positive rate when there IS an RCT-specific corruption
    tpr_by_corr={
        "arm_swap": 0.92,
        "unit_shift": 0.90,
        "timepoint_shift": 0.88,
        "row_bleed": 0.78,
        "endpoint_swap": 0.72,
        "ocr_number_confuse": 0.60,
    },
    # Base false-positive rate when extraction is actually correct
    base_fpr=0.012,
)

CORRUPTIONS = ["arm_swap","endpoint_swap","row_bleed","timepoint_shift","unit_shift","ocr_number_confuse"]

def assign_rct_corruption(rng: random.Random, scenario: tc.Scenario) -> str:
    # subtle corruptions get more weight when parser instability is higher
    subtle = {"ocr_number_confuse","row_bleed","timepoint_shift"}
    weights = []
    for c in CORRUPTIONS:
        w = 1.0
        if c in subtle:
            w *= (0.8 + 2.0*scenario.parser_instability_rate)
        if c == "arm_swap":
            w *= (0.8 + 2.5*scenario.mixing_pressure)
        weights.append(w)
    total = sum(weights)
    r = rng.random()*total
    acc=0.0
    for c,w in zip(CORRUPTIONS, weights):
        acc += w
        if r <= acc:
            return c
    return CORRUPTIONS[-1]

def rct_pack_gate(rng: random.Random, scenario: tc.Scenario, base_bundle_correct: bool) -> bool:
    """
    Return True if pack *passes*; False if it rejects.
    """
    # Adjust FPR upward with parser instability (spurious flags)
    if base_bundle_correct:
        fpr = min(0.15, PACK["base_fpr"]*(1.0 + 1.8*scenario.parser_instability_rate))
        return not (rng.random() < fpr)

    corr = assign_rct_corruption(rng, scenario)
    tpr = PACK["tpr_by_corr"].get(corr, 0.82)
    # Harder to detect when uncertainty/mixing are high
    adj_tpr = max(0.05, min(0.99, tpr*(1.0 - 0.30*scenario.mixing_pressure - 0.25*scenario.uncertainty_rate)))
    return not (rng.random() < adj_tpr)

def run_one(rng: random.Random, protocol: str) -> tc.RunOutcome:
    s = tc.Scenario(id="REAL001", domain="TC-RCT", **PAPER_PROFILE)
    policy = tc.PolicyAnchor()
    out = tc.run_protocol("TruthCert", s, policy, rng)

    if protocol == "TruthCert":
        return out

    if protocol == "TruthCert+TC-RCT":
        # only run pack if TruthCert would have shipped
        if not out.shipped:
            return out
        # Apply pack
        passes = rct_pack_gate(rng, s, base_bundle_correct=out.bundle_correct)
        if not passes:
            return tc.RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=out.tokens+PACK["token_cost"], early_termination=False)
        return tc.RunOutcome(shipped=True, bundle_correct=out.bundle_correct, rejected=False, tokens=out.tokens+PACK["token_cost"], early_termination=False)

    raise ValueError(protocol)

def summarize(n: int, seed: int):
    rng = random.Random(seed)
    rows=[]
    for protocol in ["TruthCert","TruthCert+TC-RCT"]:
        outs=[run_one(rng, protocol) for _ in range(n)]
        shipped=sum(o.shipped for o in outs)
        rejected=sum(o.rejected for o in outs)
        false_ship=sum(o.shipped and (not o.bundle_correct) for o in outs)
        correct_ship=sum(o.shipped and o.bundle_correct for o in outs)
        tokens=sum(o.tokens for o in outs)
        rows.append(dict(
            protocol=protocol,
            n=n,
            shipped_pct=shipped/n,
            reject_pct=rejected/n,
            false_ship_pct=false_ship/n,
            mean_tokens=tokens/n,
            tokens_per_correct=(tokens/correct_ship) if correct_ship else math.inf
        ))
    df=pd.DataFrame(rows)
    for c in ["shipped_pct","reject_pct","false_ship_pct"]:
        df[c]=(df[c]*100).round(2)
    df["mean_tokens"]=df["mean_tokens"].round(1)
    df["tokens_per_correct"]=df["tokens_per_correct"].round(1)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args=ap.parse_args()
    df=summarize(args.n, args.seed)
    print(df.to_string(index=False))
    print("\nInterpretation:")
    print("- TC-RCT tends to reduce *false_ship* by catching RCT-specific corruptions.")
    print("- It can also increase *reject_pct* even on correct bundles (false positives).")
    print("- Whether that's 'value' depends on your objective function (safety vs throughput vs cost).")

if __name__=="__main__":
    main()

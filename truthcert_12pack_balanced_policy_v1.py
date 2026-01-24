#!/usr/bin/env python3
"""
truthcert_12pack_balanced_policy_v1.py

Adds a *Balanced* pack-execution policy on top of:
  - TruthCert (baseline)
  - TruthCert+Packs (always run the domain pack gate)

Balanced here means:
  - Always run packs for "high risk" scenarios,
  - Often run packs for "medium risk",
  - Sometimes run packs for "low risk".

This is still a *toy* (synthetic) benchmark. It's useful to:
  - demonstrate the safety/throughput/cost tradeoff,
  - and tune a pack-routing policy before moving to real task suites.

Run:
  python truthcert_12pack_balanced_policy_v1.py --n 3000 --seed 2026
"""
import argparse, random, math
from pathlib import Path
import importlib.util, sys
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE / "truthcert_12pack_benchmark_v1.py"

spec = importlib.util.spec_from_file_location("tc12bench", BENCH)
tc12 = importlib.util.module_from_spec(spec)
sys.modules["tc12bench"] = tc12
spec.loader.exec_module(tc12)

tc = tc12.tc
gen_rich_scenarios = tc12.gen_rich_scenarios
assign_corruption_family = tc12.assign_corruption_family
PACK_VALIDATOR_POWER = tc12.PACK_VALIDATOR_POWER
PACK_VALIDATOR_TOKENS = tc12.PACK_VALIDATOR_TOKENS

def risk_score(sc):
    b = sc.base
    return 0.30*b.corruption_rate + 0.30*b.parser_instability_rate + 0.25*b.mixing_pressure + 0.15*b.uncertainty_rate

def pack_gate(base_out, sc):
    if base_out.rejected or not base_out.shipped:
        return base_out

    pack = sc.base.domain
    power = PACK_VALIDATOR_POWER[pack]
    tokens = PACK_VALIDATOR_TOKENS[pack]

    if base_out.bundle_correct:
        fpr = power["base_fpr"]
        adj_fpr = min(0.10, fpr*(1.0 + 1.5*sc.base.parser_instability_rate))
        rng_pack = random.Random((hash((sc.base.id, "pack_fpr", pack)) & 0xFFFFFFFF))
        reject = rng_pack.random() < adj_fpr
    else:
        corr = assign_corruption_family(sc, "TruthCert")
        tpr = power["per_corr"].get(corr, power["base_tpr"])
        adj_tpr = max(0.05, min(0.99, tpr*(1.0 - 0.35*sc.base.mixing_pressure - 0.25*sc.base.uncertainty_rate)))
        rng_pack = random.Random((hash((sc.base.id, "pack_tpr", pack, corr)) & 0xFFFFFFFF))
        reject = rng_pack.random() < adj_tpr

    if reject:
        return tc.RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=base_out.tokens+tokens, early_termination=False)
    return tc.RunOutcome(shipped=True, bundle_correct=base_out.bundle_correct, rejected=False, tokens=base_out.tokens+tokens, early_termination=False)

def summarize(outs):
    n=len(outs)
    shipped=sum(o.shipped for o in outs)
    rejected=sum(o.rejected for o in outs)
    false_ship=sum((o.shipped and (not o.bundle_correct)) for o in outs)
    tokens=sum(o.tokens for o in outs)
    correct_shipped=sum((o.shipped and o.bundle_correct) for o in outs)
    return dict(
        shipped_pct=shipped/n,
        false_ship_pct=false_ship/n,
        reject_pct=rejected/n,
        mean_tokens_per_bundle=tokens/n,
        tokens_per_correct_shipped=(tokens/correct_shipped) if correct_shipped else math.inf
    )

def run(args):
    scenarios = gen_rich_scenarios(args.n, args.seed)
    policy_anchor = tc.PolicyAnchor()

    base=[]
    packs=[]
    balanced=[]
    for sc in scenarios:
        rng_base = random.Random((hash((sc.base.id, "TruthCert")) & 0xFFFFFFFF))
        b = tc.run_protocol("TruthCert", sc.base, policy_anchor, rng_base)
        base.append(b)
        packs.append(pack_gate(b, sc))

        # Balanced routing
        if b.rejected or not b.shipped:
            balanced.append(b)
            continue

        r = risk_score(sc)
        if r >= args.t_high:
            p_run = 1.0
        elif r >= args.t_low:
            p_run = args.p_mid
        else:
            p_run = args.p_low

        rng_run = random.Random((hash((sc.base.id, "pack_run_decision", args.t_low, args.t_high, args.p_low, args.p_mid)) & 0xFFFFFFFF))
        if rng_run.random() < p_run:
            balanced.append(pack_gate(b, sc))
        else:
            balanced.append(b)

    rows = [
        {"protocol":"TruthCert", **summarize(base)},
        {"protocol":"TruthCert+Packs(always)", **summarize(packs)},
        {"protocol":"TruthCert+Packs(Balanced)", **summarize(balanced)},
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nBalanced policy params:", dict(t_low=args.t_low, t_high=args.t_high, p_low=args.p_low, p_mid=args.p_mid))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--t_low", type=float, default=0.35)
    ap.add_argument("--t_high", type=float, default=0.60)
    ap.add_argument("--p_low", type=float, default=0.20)
    ap.add_argument("--p_mid", type=float, default=0.45)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()

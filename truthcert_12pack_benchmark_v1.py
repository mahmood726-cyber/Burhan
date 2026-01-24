#!/usr/bin/env python3
"""
truthcert_12pack_benchmark_v1.py

A "richer toy" benchmark that models 12 TruthCert domain packs as additional
fail-closed validators with domain-specific failure signatures ("corruption families").

This is NOT a substitute for real curated tasks. It is a scaffolding layer to:
- force explicit tradeoffs (false_ship vs reject vs cost),
- model realistic error types per domain,
- and test whether packs improve safety without wrecking throughput.

Usage (in the same folder as truthcert_toy_benchmark.py):
  python truthcert_12pack_benchmark_v1.py --n 4800 --seed 2026
"""
import argparse, random, math
from dataclasses import dataclass
from pathlib import Path
import importlib.util, sys

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / "truthcert_toy_benchmark.py"

spec = importlib.util.spec_from_file_location("truthcert_toy_benchmark", BASE)
tc = importlib.util.module_from_spec(spec)
sys.modules["truthcert_toy_benchmark"] = tc
spec.loader.exec_module(tc)

PACKS = [
    "TC-RCT","TC-SR-SCREEN","TC-OBS","TC-IPD","TC-PV","TC-DIAG",
    "TC-GRADE","TC-TRIALREG","TC-CODE","TC-ANALYSIS","TC-SCI-MODEL","TC-RESEARCH"
]

PACK_CORRUPTIONS = {
    "TC-RCT": ["arm_swap","endpoint_swap","row_bleed","timepoint_shift","unit_shift","ocr_number_confuse"],
    "TC-SR-SCREEN": ["design_misclass","criteria_swap","duplicate_injection","population_misread","outcome_misread"],
    "TC-OBS": ["adjusted_unadjusted_swap","exposure_outcome_swap","scale_error","time_origin_confusion","covariate_omission"],
    "TC-IPD": ["target_leakage","split_overlap","time_leakage","label_definition_drift","preprocessing_mismatch"],
    "TC-PV": ["suspect_concomitant_swap","seriousness_flip","onset_date_shift","event_coding_error","duplicate_case"],
    "TC-DIAG": ["cell_swap","threshold_shift","partial_verification_bias","reference_standard_mixup","spectrum_bias"],
    "TC-GRADE": ["certainty_inflation","downgrade_reason_swap","outcome_priority_swap","imprecision_misread","indirectness_miss"],
    "TC-TRIALREG": ["endpoint_switching","sample_size_swap","status_misread","timepoint_mismatch","retraction_missed"],
    "TC-CODE": ["hallucinated_api","wrong_file_patch","test_masking","nondeterminism_injection","security_regression"],
    "TC-ANALYSIS": ["silent_join_error","seed_drift","label_leakage","rounding_bias","p_hacking_pattern"],
    "TC-SCI-MODEL": ["unit_mismatch","sign_flip","non_identifiable","confounding_misspec","boundary_condition_error"],
    "TC-RESEARCH": ["citation_drift","source_mixing","cherry_pick_bias","scope_creep","hallucinated_fact"],
}

PACK_VALIDATOR_POWER = {
    "TC-RCT":       dict(base_tpr=0.82, per_corr={"arm_swap":0.92,"unit_shift":0.90,"timepoint_shift":0.88,"row_bleed":0.78,"endpoint_swap":0.72,"ocr_number_confuse":0.60}, base_fpr=0.012),
    "TC-SR-SCREEN": dict(base_tpr=0.62, per_corr={"design_misclass":0.75,"criteria_swap":0.65,"duplicate_injection":0.80,"population_misread":0.55,"outcome_misread":0.50}, base_fpr=0.020),
    "TC-OBS":       dict(base_tpr=0.70, per_corr={"adjusted_unadjusted_swap":0.85,"scale_error":0.80,"exposure_outcome_swap":0.78,"covariate_omission":0.60,"time_origin_confusion":0.55}, base_fpr=0.016),
    "TC-IPD":       dict(base_tpr=0.80, per_corr={"target_leakage":0.92,"split_overlap":0.90,"time_leakage":0.85,"label_definition_drift":0.65,"preprocessing_mismatch":0.70}, base_fpr=0.018),
    "TC-PV":        dict(base_tpr=0.60, per_corr={"seriousness_flip":0.75,"suspect_concomitant_swap":0.70,"onset_date_shift":0.65,"duplicate_case":0.80,"event_coding_error":0.45}, base_fpr=0.020),
    "TC-DIAG":      dict(base_tpr=0.85, per_corr={"cell_swap":0.92,"threshold_shift":0.90,"reference_standard_mixup":0.80,"partial_verification_bias":0.55,"spectrum_bias":0.40}, base_fpr=0.012),
    "TC-GRADE":     dict(base_tpr=0.55, per_corr={"certainty_inflation":0.65,"downgrade_reason_swap":0.55,"outcome_priority_swap":0.60,"imprecision_misread":0.45,"indirectness_miss":0.40}, base_fpr=0.025),
    "TC-TRIALREG":  dict(base_tpr=0.72, per_corr={"endpoint_switching":0.80,"sample_size_swap":0.85,"status_misread":0.75,"timepoint_mismatch":0.70,"retraction_missed":0.90}, base_fpr=0.012),
    "TC-CODE":      dict(base_tpr=0.90, per_corr={"test_masking":0.95,"wrong_file_patch":0.85,"hallucinated_api":0.88,"nondeterminism_injection":0.80,"security_regression":0.70}, base_fpr=0.010),
    "TC-ANALYSIS":  dict(base_tpr=0.78, per_corr={"silent_join_error":0.88,"seed_drift":0.75,"label_leakage":0.85,"rounding_bias":0.55,"p_hacking_pattern":0.45}, base_fpr=0.015),
    "TC-SCI-MODEL": dict(base_tpr=0.72, per_corr={"unit_mismatch":0.90,"sign_flip":0.80,"non_identifiable":0.65,"boundary_condition_error":0.60,"confounding_misspec":0.55}, base_fpr=0.020),
    "TC-RESEARCH":  dict(base_tpr=0.55, per_corr={"citation_drift":0.80,"source_mixing":0.75,"hallucinated_fact":0.55,"cherry_pick_bias":0.40,"scope_creep":0.35}, base_fpr=0.025),
}

PACK_VALIDATOR_TOKENS = {
    "TC-RCT": 260, "TC-SR-SCREEN": 160, "TC-OBS": 190, "TC-IPD": 320, "TC-PV": 170, "TC-DIAG": 210,
    "TC-GRADE": 160, "TC-TRIALREG": 220, "TC-CODE": 380, "TC-ANALYSIS": 320, "TC-SCI-MODEL": 300, "TC-RESEARCH": 140
}

@dataclass(frozen=True)
class RichScenario:
    base: tc.Scenario

def gen_rich_scenarios(n:int, seed:int):
    rng = random.Random(seed)
    scenarios=[]
    for i in range(n):
        pack = rng.choice(PACKS)
        if pack in ["TC-RCT","TC-DIAG"]:
            corruption_rate = tc.beta(rng, 2.3, 4.6)
            parser = tc.beta(rng, 2.0, 4.3)
            mixing = tc.beta(rng, 1.8, 5.2)
            uncert = tc.beta(rng, 2.0, 5.3)
            n_fields = rng.randint(55, 130)
            n_crit = rng.randint(12, 26)
        elif pack in ["TC-CODE"]:
            corruption_rate = tc.beta(rng, 1.9, 5.8)
            parser = tc.beta(rng, 1.1, 8.0)
            mixing = tc.beta(rng, 2.4, 4.8)
            uncert = tc.beta(rng, 1.5, 6.5)
            n_fields = rng.randint(45, 150)
            n_crit = rng.randint(10, 22)
        elif pack in ["TC-IPD","TC-SCI-MODEL"]:
            corruption_rate = tc.beta(rng, 2.0, 5.5)
            parser = tc.beta(rng, 1.4, 6.6)
            mixing = tc.beta(rng, 2.7, 4.3)
            uncert = tc.beta(rng, 2.7, 4.2)
            n_fields = rng.randint(80, 180)
            n_crit = rng.randint(15, 32)
        elif pack in ["TC-GRADE","TC-RESEARCH"]:
            corruption_rate = tc.beta(rng, 1.6, 6.3)
            parser = tc.beta(rng, 1.2, 7.4)
            mixing = tc.beta(rng, 2.9, 4.0)
            uncert = tc.beta(rng, 2.8, 4.1)
            n_fields = rng.randint(35, 110)
            n_crit = rng.randint(10, 22)
        else:
            corruption_rate = tc.beta(rng, 1.8, 6.0)
            parser = tc.beta(rng, 1.6, 6.0)
            mixing = tc.beta(rng, 2.2, 5.0)
            uncert = tc.beta(rng, 2.1, 5.0)
            n_fields = rng.randint(35, 140)
            n_crit = rng.randint(8, 26)

        scenarios.append(RichScenario(base=tc.Scenario(
            id=f"RS{i:05d}",
            domain=pack,
            n_fields=n_fields,
            n_critical_fields=n_crit,
            corruption_rate=corruption_rate,
            parser_instability_rate=parser,
            mixing_pressure=mixing,
            uncertainty_rate=uncert
        )))
    return scenarios

def assign_corruption_family(s: RichScenario, tag:str):
    rng = random.Random((hash((s.base.id, "corr", tag)) & 0xFFFFFFFF))
    fams = PACK_CORRUPTIONS[s.base.domain]
    p = s.base.parser_instability_rate
    m = s.base.mixing_pressure
    subtle = {"ocr_number_confuse","population_misread","outcome_misread","time_origin_confusion","event_coding_error",
              "partial_verification_bias","spectrum_bias","indirectness_miss","p_hacking_pattern","confounding_misspec",
              "cherry_pick_bias","scope_creep"}
    weights=[]
    for f in fams:
        weights.append(0.4 + 1.6*(p+m)/2 if f in subtle else 1.0)
    tot=sum(weights)
    r=rng.random()*tot
    acc=0.0
    for f,w in zip(fams,weights):
        acc += w
        if r <= acc: return f
    return fams[-1]

def run_truthcert_plus_packs(s: RichScenario):
    policy = tc.PolicyAnchor()
    rng_base = random.Random((hash((s.base.id, "TruthCert")) & 0xFFFFFFFF))
    base = tc.run_protocol("TruthCert", s.base, policy, rng_base)
    if base.rejected or not base.shipped:
        return base, None
    pack = s.base.domain
    power = PACK_VALIDATOR_POWER[pack]
    tokens = PACK_VALIDATOR_TOKENS[pack]

    if base.bundle_correct:
        fpr = power["base_fpr"]
        adj_fpr = min(0.10, fpr*(1.0 + 1.5*s.base.parser_instability_rate))
        rng_pack = random.Random((hash((s.base.id, "pack_fpr", pack)) & 0xFFFFFFFF))
        reject = rng_pack.random() < adj_fpr
        corr = None
    else:
        corr = assign_corruption_family(s, "TruthCert")
        tpr = power["per_corr"].get(corr, power["base_tpr"])
        adj_tpr = max(0.05, min(0.99, tpr*(1.0 - 0.35*s.base.mixing_pressure - 0.25*s.base.uncertainty_rate)))
        rng_pack = random.Random((hash((s.base.id, "pack_tpr", pack, corr)) & 0xFFFFFFFF))
        reject = rng_pack.random() < adj_tpr

    if reject:
        return tc.RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=base.tokens+tokens, early_termination=False), corr
    return tc.RunOutcome(shipped=True, bundle_correct=base.bundle_correct, rejected=False, tokens=base.tokens+tokens, early_termination=False), corr

def run_protocol(protocol:str, s: RichScenario):
    policy = tc.PolicyAnchor()
    rng = random.Random((hash((s.base.id, protocol)) & 0xFFFFFFFF))
    out = tc.run_protocol(protocol, s.base, policy, rng)
    corr = assign_corruption_family(s, protocol) if out.shipped and (not out.bundle_correct) else None
    return out, corr

def summarize(protocol:str, scenarios):
    outs=[]
    corr_counts={}
    for s in scenarios:
        if protocol=="TruthCert+Packs":
            out, corr = run_truthcert_plus_packs(s)
        else:
            out, corr = run_protocol(protocol, s)
        outs.append(out)
        if corr is not None: corr_counts[corr]=corr_counts.get(corr,0)+1
    n=len(outs)
    shipped=sum(o.shipped for o in outs)
    rejected=sum(o.rejected for o in outs)
    false_ship=sum((o.shipped and (not o.bundle_correct)) for o in outs)
    tokens=sum(o.tokens for o in outs)
    correct_shipped=sum((o.shipped and o.bundle_correct) for o in outs)
    return {
        "protocol": protocol,
        "n_scenarios": n,
        "shipped_pct": shipped/n,
        "false_ship_pct": false_ship/n,
        "reject_pct": rejected/n,
        "mean_tokens_per_bundle": tokens/n,
        "tokens_per_correct_shipped": (tokens/correct_shipped) if correct_shipped else math.inf,
        "wrong_shipped_count": false_ship,
        "wrong_ship_top3": sorted(corr_counts.items(), key=lambda kv: kv[1], reverse=True)[:3],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4800)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    scenarios = gen_rich_scenarios(args.n, args.seed)
    protocols = ["SinglePass","SchemaOnly","VoteOnly","VoteOnly_TCParser","TruthCert","TruthCert+Packs"]
    df = pd.DataFrame([summarize(p, scenarios) for p in protocols])
    print(df[["protocol","shipped_pct","false_ship_pct","reject_pct","mean_tokens_per_bundle","tokens_per_correct_shipped","wrong_shipped_count","wrong_ship_top3"]].to_string(index=False))

if __name__ == "__main__":
    main()

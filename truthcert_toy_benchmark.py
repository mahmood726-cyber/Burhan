"""
TruthCert v3.1 — Toy Monte Carlo benchmark harness (Simulation Contract v1-shaped)

This is NOT an implementation of TruthCert. It's a *smoke-test model* that helps you
stress the gate logic and compare trade-offs against simpler specs under explicit
assumptions about model error, parsing instability, mixing pressure, etc.

Why "toy"? Real performance requires running the protocol on real documents
with real models and ground-truth labels. This file exists to:
  - reproduce the quick comparison table,
  - let you tweak assumptions and see how the metrics move,
  - generate a Contract-v1-shaped summary report.

Run:
  python truthcert_toy_benchmark.py
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# Simulation objects
# ----------------------------
@dataclass(frozen=True)
class Scenario:
    id: str
    domain: str
    n_fields: int
    n_critical_fields: int
    corruption_rate: float
    parser_instability_rate: float
    mixing_pressure: float
    uncertainty_rate: float


@dataclass(frozen=True)
class PolicyAnchor:
    fact_agreement: float = 0.80
    interpretation_agreement: float = 0.70
    blindspot_r: float = 0.60
    material_disagreement_pct: float = 0.05


@dataclass
class RunOutcome:
    shipped: bool
    bundle_correct: bool
    rejected: bool
    tokens: int
    early_termination: bool


# ----------------------------
# Scenario generator
# ----------------------------
def beta(rng: random.Random, a: float, b: float) -> float:
    return rng.betavariate(a, b)


def gen_scenarios(n: int, seed: int = 42) -> List[Scenario]:
    rng = random.Random(seed)
    domains = ["clinical_trials", "observational", "meta_analysis", "other"]
    scenarios = []
    for i in range(n):
        domain = rng.choice(domains)
        complexity = rng.random()
        if complexity < 0.35:
            n_fields = rng.randint(15, 35)
            n_crit = rng.randint(5, 10)
        elif complexity < 0.75:
            n_fields = rng.randint(35, 70)
            n_crit = rng.randint(8, 16)
        else:
            n_fields = rng.randint(70, 130)
            n_crit = rng.randint(12, 24)

        # Rates in [0,1], skewed toward "mostly OK" but with a tail of nasty cases
        corruption = beta(rng, 1.6, 6.0)
        parser_instability = beta(rng, 1.4, 6.5)
        mixing = beta(rng, 1.8, 5.5)
        uncertainty = beta(rng, 1.8, 6.2)

        scenarios.append(
            Scenario(
                id=f"S{i:04d}",
                domain=domain,
                n_fields=n_fields,
                n_critical_fields=n_crit,
                corruption_rate=corruption,
                parser_instability_rate=parser_instability,
                mixing_pressure=mixing,
                uncertainty_rate=uncertainty,
            )
        )
    return scenarios


# ----------------------------
# Witness model
# ----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def witness_outputs(
    rng: random.Random,
    truth: np.ndarray,
    scenario: Scenario,
    family: str,
    base_acc_by_family: Dict[str, float],
    blindspot_event: bool,
    blindspot_fields: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    base_acc = base_acc_by_family.get(family, 0.90)
    p_correct = clamp01(
        base_acc
        - 0.35 * scenario.corruption_rate
        - 0.25 * scenario.parser_instability_rate
        - 0.30 * scenario.mixing_pressure
        - 0.10 * scenario.uncertainty_rate
    )

    arm_swap = rng.random() < (0.03 + 0.20 * scenario.mixing_pressure)
    provenance_valid = rng.random() < clamp01(1.0 - 0.60 * scenario.mixing_pressure - 0.20 * scenario.uncertainty_rate)
    mixing_suspicion = rng.random() < (0.50 * scenario.mixing_pressure)

    vals = np.empty_like(truth)
    for j, t in enumerate(truth):
        if blindspot_event and blindspot_fields[j] and family == "A":
            vals[j] = t * (1.15 + rng.normalvariate(0, 0.01))
            continue

        is_correct = rng.random() < p_correct
        if is_correct:
            vals[j] = t * (1.0 + rng.normalvariate(0, 0.003))
        else:
            scale = 0.20 + 0.60 * scenario.corruption_rate + 0.25 * scenario.mixing_pressure
            vals[j] = t * (1.0 + rng.normalvariate(0, scale))
            if rng.random() < 0.05:
                vals[j] *= 10.0
            if rng.random() < 0.03:
                vals[j] *= -1.0

        if arm_swap:
            vals[j] = vals[j] * (1.0 + rng.normalvariate(0, 0.10))

    flags = {
        "arm_swap": arm_swap,
        "provenance_valid": provenance_valid,
        "mixing_suspicion": mixing_suspicion,
    }
    return vals, flags


def is_field_correct(extracted: float, truth: float) -> bool:
    denom = max(abs(truth), 1e-12)
    return abs(extracted - truth) / denom <= 0.01


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


# ----------------------------
# Gate logic (toy approximation)
# ----------------------------
def gate_parser_arbitration(rng: random.Random, scenario: Scenario) -> Tuple[bool, bool, int]:
    parse_unstable = rng.random() < scenario.parser_instability_rate
    tokens = 0
    if not parse_unstable:
        return True, False, tokens
    tokens += 120
    material_disagree = rng.random() < (0.50 * scenario.parser_instability_rate)
    return (not material_disagree), True, tokens


def bundle_agreement(values: np.ndarray, policy: PolicyAnchor) -> float:
    consensus = np.median(values, axis=0)
    tol = policy.material_disagreement_pct
    abs_floor = 1e-12
    agree = np.abs(values - consensus) <= np.maximum(tol * np.abs(consensus), abs_floor)
    per_field = agree.mean(axis=0)
    return float(per_field.mean())


def gate_blindspot(values: np.ndarray, truth: np.ndarray, policy: PolicyAnchor) -> Tuple[bool, float]:
    n_w, n_f = values.shape
    correct = np.array([[1 if is_field_correct(values[i, j], truth[j]) else 0 for j in range(n_f)] for i in range(n_w)])
    max_r = 0.0
    for i in range(n_w):
        for k in range(i + 1, n_w):
            r = corr_safe(correct[i], correct[k])
            max_r = max(max_r, r)
            if r > policy.blindspot_r:
                return False, max_r
    return True, max_r


def gate_structural(rng: random.Random, scenario: Scenario) -> bool:
    p_fail = 0.02 + 0.30 * scenario.parser_instability_rate
    return rng.random() > p_fail


def gate_antimixing(flags_list: List[Dict[str, bool]]) -> bool:
    for fl in flags_list:
        if fl["mixing_suspicion"] or (not fl["provenance_valid"]):
            return False
    return True


def gate_adversarial(rng: random.Random, scenario: Scenario) -> bool:
    has_corruption = rng.random() < (0.85 * scenario.corruption_rate)
    if not has_corruption:
        return True
    detect = rng.random() < 0.95
    return detect


def token_cost_for_witness(n_fields: int) -> int:
    return 500 + 18 * n_fields


def run_protocol(protocol: str, scenario: Scenario, policy: PolicyAnchor, rng: random.Random) -> RunOutcome:
    truth = np.array([rng.uniform(0.1, 100.0) for _ in range(scenario.n_critical_fields)], dtype=float)
    tokens = 0

    if protocol in ["TruthCert", "VoteOnly_TCParser"]:
        ok, _, t = gate_parser_arbitration(rng, scenario)
        tokens += t
        if not ok:
            return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)

    if protocol == "SinglePass":
        families = ["A"]
        do_blindspot = False
        do_antimixing = False
        do_adversarial = False
        do_structural = True
        do_agreement_gate = False
    elif protocol == "SchemaOnly":
        families = ["A"]
        do_blindspot = False
        do_antimixing = False
        do_adversarial = False
        do_structural = True
        do_agreement_gate = False
    elif protocol == "VoteOnly":
        families = ["A", "A", "A"]
        do_blindspot = False
        do_antimixing = False
        do_adversarial = False
        do_structural = True
        do_agreement_gate = True
    elif protocol == "VoteOnly_TCParser":
        families = ["A", "A", "A"]
        do_blindspot = False
        do_antimixing = False
        do_adversarial = False
        do_structural = True
        do_agreement_gate = True
    elif protocol == "TruthCert":
        families = ["A", "A", "B"]  # approximates heterogeneity
        do_blindspot = True
        do_antimixing = True
        do_adversarial = True
        do_structural = True
        do_agreement_gate = True
    else:
        raise ValueError(protocol)

    blindspot_event = rng.random() < (0.12 + 0.35 * scenario.corruption_rate)
    blindspot_fields = np.array([rng.random() < 0.30 for _ in range(scenario.n_critical_fields)], dtype=bool)
    base_acc = {"A": 0.92, "B": 0.90}

    vals_list, flags_list = [], []
    for fam in families:
        vals, fl = witness_outputs(rng, truth, scenario, fam, base_acc, blindspot_event, blindspot_fields)
        vals_list.append(vals)
        flags_list.append(fl)
        tokens += token_cost_for_witness(scenario.n_fields)

    vals_arr = np.vstack(vals_list)

    if do_structural and (not gate_structural(rng, scenario)):
        return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)

    if do_antimixing and (not gate_antimixing(flags_list)):
        return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)

    if do_blindspot:
        ok, _ = gate_blindspot(vals_arr, truth, policy)
        if not ok:
            return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)

    if do_agreement_gate:
        agree = bundle_agreement(vals_arr, policy)
        if protocol == "TruthCert":
            if 0.70 <= agree < policy.fact_agreement:
                helper_family = "C" if "C" in families else ("B" if "B" not in families else "C")
                base_acc[helper_family] = 0.91
                helper_vals, helper_flags = witness_outputs(rng, truth, scenario, helper_family, base_acc, blindspot_event, blindspot_fields)
                tokens += token_cost_for_witness(scenario.n_fields)
                vals_arr = np.vstack([vals_arr, helper_vals])
                flags_list.append(helper_flags)
                agree = bundle_agreement(vals_arr, policy)

            if agree < policy.fact_agreement:
                return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)
        else:
            if agree < policy.fact_agreement:
                return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)

    if do_adversarial:
        tokens += 350
        if not gate_adversarial(rng, scenario):
            return RunOutcome(shipped=False, bundle_correct=False, rejected=True, tokens=tokens, early_termination=False)

    consensus = np.median(vals_arr, axis=0)
    all_fields_correct = all(is_field_correct(consensus[j], truth[j]) for j in range(scenario.n_critical_fields))
    no_arm_swap = not any(fl["arm_swap"] for fl in flags_list)
    provenance_valid = all(fl["provenance_valid"] for fl in flags_list)
    bundle_correct = all_fields_correct and no_arm_swap and provenance_valid

    return RunOutcome(shipped=True, bundle_correct=bundle_correct, rejected=False, tokens=tokens, early_termination=False)


def summarize(protocol: str, scenarios: List[Scenario], seed: int = 1) -> Dict[str, float]:
    rng = random.Random(seed)
    policy = PolicyAnchor()
    outcomes = [run_protocol(protocol, sc, policy, rng) for sc in scenarios]

    n = len(outcomes)
    shipped = sum(1 for o in outcomes if o.shipped)
    rejected = sum(1 for o in outcomes if o.rejected)
    correct_shipped = sum(1 for o in outcomes if o.shipped and o.bundle_correct)
    false_shipped = sum(1 for o in outcomes if o.shipped and (not o.bundle_correct))
    tokens_total = sum(o.tokens for o in outcomes)

    return {
        "protocol": protocol,
        "n_scenarios": n,
        "shipped_pct": shipped / n,
        "reject_pct": rejected / n,
        "false_ship_pct": false_shipped / n,
        "mean_tokens_per_bundle": tokens_total / n,
        "tokens_per_correct_shipped": (tokens_total / correct_shipped) if correct_shipped else float("inf"),
        "early_termination_rate": 0.0,
    }


def main() -> None:
    scenarios = gen_scenarios(1000, seed=2026)
    protocols = ["SinglePass", "SchemaOnly", "VoteOnly", "VoteOnly_TCParser", "TruthCert"]
    rows = [summarize(p, scenarios, seed=7) for p in protocols]
    df = pd.DataFrame(rows)
    pct_cols = ["shipped_pct", "reject_pct", "false_ship_pct", "early_termination_rate"]
    df[pct_cols] = (df[pct_cols] * 100).round(1)
    df["mean_tokens_per_bundle"] = df["mean_tokens_per_bundle"].round(0).astype(int)
    df["tokens_per_correct_shipped"] = df["tokens_per_correct_shipped"].round(0).astype(int)
    print(df.to_string(index=False))

    # Contract-v1-shaped report for the last protocol (TruthCert)
    tc = next(r for r in rows if r["protocol"] == "TruthCert")
    report = {
        "n_scenarios": 1000,
        "workload_mix": {},  # fill with your real domain/complexity mix when you build a real suite
        "mode": "fixed",
        "heterogeneity": "required (toy approx)",
        "budget_enforcement": "off",
        "shipped_pct": tc["shipped_pct"],
        "false_ship_pct": tc["false_ship_pct"],
        "reject_pct": tc["reject_pct"],
        "mean_tokens_per_bundle": tc["mean_tokens_per_bundle"],
        "tokens_per_correct_shipped": tc["tokens_per_correct_shipped"],
        "early_termination_rate": tc["early_termination_rate"],
        "reproducibility": {"generator": "toy", "seed": 2026},
    }
    print("\nSimulation-report (Contract v1-shaped):")
    print(report)


if __name__ == "__main__":
    main()

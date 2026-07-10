# TruthCert 

**TruthCert is a certification protocol for LLM outputs** in high-stakes workflows (starting with evidence extraction and research).  
It is **not** a model. It is a **fail-closed verification + disclosure standard**: outputs are only allowed to “ship” when they meet a published policy and can be audited.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.18363659

---

## Why TruthCert exists

In high-stakes use, the worst failure mode is not “the model makes mistakes.”  
It’s **quietly wrong outputs that look confident** (e.g., arm swaps, timepoint drift, unit errors, silent regressions in code, citation drift in synthesis).

TruthCert’s core idea is simple:

> Don’t ask “does this look right?”  
> Ask “is this **certified** under a published policy, with auditable evidence — or rejected?”

---

## What “Certified” means (high level)

A **TruthCert-CERTIFIED** bundle is an output that:
- is **scope-locked** (no drift beyond the Scope Lock),
- includes **provenance per atomic value** (one chain per value),
- passes **multi-witness verification** (≥3 independent witnesses) with arbitration,
- passes **versioned validator checks** (domain pack validators),
- is recorded as an immutable artifact (bundle hash, ledger reference),
- includes required **disclosures** (witness mode/count, heterogeneity, external checks, budget mode, validator version, etc.).

TruthCert is intentionally **fail-closed**: if verification is insufficient, the output is **REJECTED**.

---

## Repo layout

All scripts and documents live at the repository root (flat layout):

- `TruthCert_v3.1.0-FINAL_Public_Frozen.md` — the frozen public spec
- `TruthCert_12_Domain_Extensions_v1.md` — the 12 domain extension packs
- `TruthCert_Certification_Badge_Disclosure_Template_v1.md` — copy/paste badge + disclosure blocks
- `truthcert_toy_benchmark.py` — toy simulation harness
- `truthcert_12pack_benchmark_v1.py`, `truthcert_12pack_balanced_policy_v1.py` — 12-pack benchmarks
- `simulate_truthcert_vs_rct_pack.py` — TruthCert vs TC-RCT stress model
- `score_contract_v1.py` — scoring script for Contract-v1 metrics
- `tc_pcn.py`, `tc_prm.py`, `tc_qwed.py`, `tc_formal.py`, `tc_rivals.py`, `tc_verifier.py`, `tc_integration.py` — verification extension modules
- `tests/` — pytest suite

---

## Quick start (simulation harness)

Run the scripts from the repository root:

```bash
python truthcert_toy_benchmark.py
python truthcert_12pack_benchmark_v1.py --n 4800 --seed 2026
python truthcert_12pack_balanced_policy_v1.py
python simulate_truthcert_vs_rct_pack.py --n 5000 --seed 2026
```

# TruthCert Domain Extensions (12 Packs)
**Date:** 2026-01-24  
**Purpose:** These are **domain-specific “extensions” (packs/profiles)** that optimize TruthCert for high-stakes scientific and professional LLM workflows **without changing the frozen core**.

Each pack defines:
- a **certified payload** (comparable leaf keys, not giant blobs),
- **critical keys** (catastrophic-if-wrong),
- a **validator set** (machine-checkable rules mapped to TruthCert gates),
- **corruption registry extensions** (adversarial traps and failure modes),
- and a **scenario pack** approach compatible with Simulation Contract v1.

> Design rule: Certify **checkable, comparable leaf keys** (numbers, enums, short strings, hashes, pointers).  
> Keep large artifacts (PDFs, patches, notebooks) as **attachments**, and certify **their hashes + outcomes**.

---

## How packs connect to the TruthCert core
You keep the core invariant gates (A-lane exploration, B-lane verification) unchanged. Packs “plug in” via:
1) **Expected schema**: pack payload schema defines what “structural validity” means.  
2) **Validator set (versioned)**: pack provides validator IDs and rules; bundles record the validator version/hash.  
3) **Corruption registry extensions**: pack adds domain-specific corruptions (still mapped onto the core corruption families).  
4) **Benchmarking**: packs ship scenario lists and report the Contract-v1 metrics:
   - shipped_pct, false_ship_pct, reject_pct  
   - mean_tokens_per_bundle, tokens_per_correct_shipped  
   - early_termination_rate

---

## Shared conventions used below

### Provenance pointers (pack-neutral)
Every certified bundle should include `provenance.ptr[]` entries that can resolve to:
- PDF: `{doc_hash, page, table_id, row, col, bbox}`
- Code: `{repo_commit, file_path, line_start, line_end}`
- Dataset: `{dataset_hash, file, row_ids/filters}`
- Web/doc: `{source_hash, quote_span}`

### Gate mapping shorthand
- **B3**: structural validity, schema, bounds, pointer resolvability, basic checks  
- **B5**: semantic agreement across witnesses on leaf keys  
- **B8**: adversarial corruptions (different-family generator) must be caught  
- **B10**: optional external checks (e.g., registry/retraction lookups)  
- **B11**: efficiency accounting

---

# Pack 1 — TC-RCT (RCT data extraction from PDFs)
**Domain:** `RCT_EXTRACT`  
**Goal:** Extract outcome data from RCT PDFs with **arm / endpoint / timepoint / unit integrity**.

## Certified payload (suggested)
```json
{
  "study_id": "...",
  "arms": [{"name":"Tx","n":123}, {"name":"Ctrl","n":120}],
  "outcomes": [
    {
      "endpoint": "MACE",
      "timepoint": "12mo",
      "measure": "RR|OR|HR|mean_diff|events",
      "unit": "ratio|%|mg/dL|...",
      "value": 0.85,
      "se": 0.07,
      "ci95": {"low":0.74,"high":0.98},
      "provenance": {"ptr":[...]}
    }
  ]
}
```

### Critical keys (catastrophic if wrong)
- arm names + arm Ns  
- endpoint + timepoint  
- unit + value  
- provenance pointers

## Validators (starter set)
- **V-RCT-PTR (B3):** all table/text pointers resolve (page/table/cell/quote span)  
- **V-RCT-ARMALIGN (B3):** intervention/control alignment consistent across the document  
- **V-RCT-TIMEPOINT (B3):** extracted timepoint matches stated follow-up window  
- **V-RCT-UNITNORM (B3):** units normalized; conversions recorded  
- **V-RCT-ARITH (B3):** internal arithmetic checks (totals, subgroup sums, consistency) where applicable  
- **V-RCT-AGREE (B5):** witnesses agree on critical keys at strict threshold  
- **V-RCT-ADVERSARIAL (B8):** traps for arm swap, endpoint swap, row bleed, timepoint shift, unit error

## Corruption registry extensions
- `arm_swap` (Tx↔Ctrl) — **high**  
- `endpoint_swap` (primary↔secondary) — **high**  
- `row_bleed` (wrong row right column) — **high**  
- `timepoint_shift` (30d↔12mo) — **high**  
- `unit_shift` (mg↔mcg, %↔fraction) — **high**

## Scenario pack guidance
- Use real PDFs (or synthetic tables with OCR artifacts).  
- Raise `parser_instability_rate` for scans/low-quality PDFs; raise `mixing_pressure` for multiple outcomes/timepoints.  
- Gold truth must include **table cell locations** and final extracted values.

---

# Pack 2 — TC-SR-SCREEN (Systematic review screening + PRISMA logs)
**Domain:** `SR_SCREENING`  
**Goal:** Title/abstract/full-text screening that is reproducible, auditable, and taxonomy-consistent.

## Certified payload
```json
{
  "record_id": "...",
  "decision": "include|exclude|maybe",
  "exclusion_reason": "Not_RCT|Wrong_population|Wrong_outcome|...",
  "picos": {
    "population": "...",
    "intervention": "...",
    "comparator": "...",
    "outcome": "...",
    "design": "RCT|cohort|case-control|..."
  },
  "provenance": {"ptr":[...]}
}
```

### Critical keys
- decision  
- exclusion_reason (from allowed taxonomy)  
- design classification  
- provenance pointers

## Validators
- **V-SR-TAXON (B3):** exclusion_reason ∈ approved taxonomy  
- **V-SR-DUP (B3):** duplicate/near-duplicate detection + merge log  
- **V-SR-AGREE (B5):** multi-witness agreement on include/exclude and key PICOS fields  
- **V-SR-ADVERSARIAL (B8):** criteria swaps; design misclassification traps

## Corruptions
- `criteria_swap` (PICOS misread)  
- `design_misclass` (RCT↔observational)  
- `duplicate_injection` (same trial as multiple records)

## Scenario guidance
- Include borderline abstracts; include registry records + publications to push mixing pressure.

---

# Pack 3 — TC-OBS (Observational effects extraction + confounding hygiene)
**Domain:** `OBS_EFFECTS`  
**Goal:** Extract observational effect estimates with correct **adjustment status**, scale, and covariate set.

## Certified payload
```json
{
  "study_id":"...",
  "exposure":"...",
  "outcome":"...",
  "effect_type":"HR|OR|RR|beta",
  "estimate":1.22,
  "ci95":{"low":1.05,"high":1.43},
  "adjusted":true,
  "covariates":["age","sex","..."],
  "timepoint":"...",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- exposure/outcome alignment  
- effect_type + estimate + scale correctness  
- adjusted flag  
- timepoint  
- provenance

## Validators
- **V-OBS-ADJFLAG (B3):** adjusted flag matches model described in paper  
- **V-OBS-SCALE (B3):** log-scale vs raw-scale consistency; CI bounds valid  
- **V-OBS-COVREQ (B3):** covariate list present iff adjusted=true  
- **V-OBS-AGREE (B5):** agreement on core estimate + adjusted flag  
- **V-OBS-ADVERSARIAL (B8):** adjusted↔unadjusted swap; exposure/outcome swap

## Corruptions
- `adjusted_unadjusted_swap` (high)  
- `exposure_outcome_swap` (high)  
- `time_origin_confusion` (medium-high)

---

# Pack 4 — TC-IPD (IPD pipelines: cohort, splits, leakage)
**Domain:** `IPD_PIPELINE`  
**Goal:** Certify IPD pipeline steps and outputs with strong leakage protections and reproducibility.

## Certified payload
```json
{
  "dataset_hash":"...",
  "cohort":{"criteria":"...", "n": 4821},
  "split":{"method":"patient-level", "seed":1234, "train_n":..., "test_n":...},
  "label_def":"...",
  "feature_defs":["..."],
  "primary_model":"...",
  "primary_metric":{"name":"cindex","value":0.71},
  "uncertainty":{"type":"bootstrap","level":0.95,"summary":{...}},
  "env_hash":"...",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- dataset_hash, cohort.n  
- split.method + seed + integrity  
- primary_metric + uncertainty present  
- provenance

## Validators
- **V-IPD-LEAK (B3):** leakage checks pass (label leakage, post-treatment features, peeking)  
- **V-IPD-SPLIT (B3):** no subject overlap; time-aware split if required  
- **V-IPD-RERUN (B3):** rerun reproduces primary metrics within tolerance  
- **V-IPD-AGREE (B5):** agreement on cohort size + metrics  
- **V-IPD-ADVERSARIAL (B8):** leakage trap; treatment inversion; censoring mishandling

## Corruptions
- `target_leakage_trap` (high)  
- `treatment_indicator_inversion` (high)  
- `time_origin_error` (high)

---

# Pack 5 — TC-PV (Pharmacovigilance ICSR extraction)
**Domain:** `PHARMACOVIGILANCE`  
**Goal:** Extract adverse-event case details with correct timeline + seriousness classification.

## Certified payload
```json
{
  "case_id":"...",
  "suspect_drug":"...",
  "dose":"...",
  "route":"...",
  "indication":"...",
  "event":"...",
  "onset_date":"YYYY-MM-DD",
  "seriousness":"serious|non-serious",
  "dechallenge":"yes|no|unknown",
  "rechallenge":"yes|no|unknown",
  "causality":"certain|probable|possible|unlikely|unassessable",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- suspect_drug, event  
- onset_date + seriousness  
- provenance pointers

## Validators
- **V-PV-TIMELINE (B3):** timeline consistency (onset after exposure start; dechallenge plausibility)  
- **V-PV-MEDDRA (B3):** event coding sanity (if mapped)  
- **V-PV-DUP (B3):** duplicate case detection (same narrative, dates, drug)  
- **V-PV-ADVERSARIAL (B8):** suspect↔concomitant swap; seriousness flip

## Corruptions
- `suspect_concomitant_swap` (high)  
- `seriousness_flip` (high)  
- `onset_date_shift` (high)

---

# Pack 6 — TC-DIAG (Diagnostic test accuracy / 2×2)
**Domain:** `DIAGNOSTIC_DTA`  
**Goal:** Extract 2×2 tables and thresholds reliably.

## Certified payload
```json
{
  "study_id":"...",
  "index_test":"...",
  "reference_standard":"...",
  "threshold":"...",
  "tp": 88, "fp": 12, "fn": 20, "tn": 140,
  "prevalence": 0.35,
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- threshold  
- tp/fp/fn/tn  
- provenance pointers

## Validators
- **V-DTA-ARITH (B3):** 2×2 arithmetic consistency and non-negativity  
- **V-DTA-THRESH (B3):** threshold matches paper/protocol  
- **V-DTA-AGREE (B5):** witnesses converge on threshold + 2×2 cells  
- **V-DTA-ADVERSARIAL (B8):** cell swaps; threshold shifts

## Corruptions
- `cell_swap` (tp↔fp etc.) (high)  
- `threshold_shift` (high)  
- `partial_verification_trap` (medium-high)

---

# Pack 7 — TC-GRADE (GRADE + Evidence-to-Decision)
**Domain:** `GRADE_ETD`  
**Goal:** Certify certainty judgments and recommendation logic linked to extracted effects.

## Certified payload
```json
{
  "outcome":"...",
  "effect_summary":"...",
  "certainty":"high|moderate|low|very_low",
  "downgrade_reasons":["risk_of_bias","inconsistency", "..."],
  "upgrade_reasons":["large_effect","dose_response", "..."],
  "recommendation":"strong_for|conditional_for|conditional_against|strong_against",
  "notes_short":"...",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- outcome  
- certainty  
- recommendation  
- provenance

## Validators
- **V-GRADE-RULES (B3):** downgrade/upgrade reasons consistent with rule mapping  
- **V-GRADE-LINK (B3):** effect_summary matches extracted effects  
- **V-GRADE-AGREE (B5):** agreement on certainty + recommendation  
- **V-GRADE-ADVERSARIAL (B8):** certainty inflation; outcome priority swap

## Corruptions
- `certainty_inflation` (high)  
- `downgrade_reason_swap` (medium)  
- `outcome_priority_swap` (medium)

---

# Pack 8 — TC-TRIALREG (Trial registry ↔ paper reconciliation)
**Domain:** `TRIAL_REGISTRY_RECON`  
**Goal:** Detect discrepancies between registry and publication; optionally hard-stop on retractions.

## Certified payload
```json
{
  "trial_id":"NCT...",
  "registry":{"endpoints":[...],"sample_size":..., "status":"..."},
  "paper":{"endpoints":[...],"sample_size":..., "primary_timepoint":"..."},
  "mismatches":[{"type":"endpoint","detail":"..."}],
  "retraction_status":"retracted|not_retracted|unknown",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- trial_id  
- mismatches (if any)  
- retraction_status  
- provenance

## Validators
- **V-REG-DIFF (B3):** mismatch detection correctness (endpoints, n, timepoints)  
- **V-REG-RETR (B10/B3):** if external refs enabled, reject if retracted  
- **V-REG-AGREE (B5):** agreement on mismatch types  
- **V-REG-ADVERSARIAL (B8):** registry endpoint swap; status misread

## Corruptions
- `registry_endpoint_swap` (high)  
- `status_misread` (high)  
- `sample_size_swap` (high)

---

# Pack 9 — TC-CODE (Coding with LLMs: correctness via execution)
**Domain:** `CODE_LLM`  
**Goal:** Ship code only when **tests + hidden behavioral checks + determinism** pass.

## Certified payload
```json
{
  "repo_commit":"...",
  "task_type":"bugfix|feature|refactor|repro",
  "changed_files":["..."],
  "tests":{"passed":120,"failed":0,"command":"pytest -q"},
  "hidden_checks":[{"name":"no_api_break","pass":true}],
  "lint":{"pass":true},
  "typecheck":{"pass":true},
  "env_hash":"...",
  "seed_policy":"fixed|recorded|n/a",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- tests.failed == 0  
- hidden_checks all pass  
- env_hash + provenance

## Validators
- **V-CODE-CLEAN (B3):** clean-room install from lockfile succeeds  
- **V-CODE-TEST (B3):** unit/integration tests pass  
- **V-CODE-HIDDEN (B3):** hidden behavioral tests pass  
- **V-CODE-DET (B3):** deterministic run (or explicit waiver)  
- **V-CODE-AGREE (B5):** agreement on leaf-key outcomes (not identical patch)  
- **V-CODE-ADVERSARIAL (B8):** hallucinated API; wrong file patch; test masking; perf regression

## Corruptions
- `hallucinated_api` (high)  
- `wrong_file_patch` (high)  
- `test_masking` (high)  
- `nondeterminism_injection` (high)  
- `perf_regression` (medium-high)

---

# Pack 10 — TC-ANALYSIS (Statistical analysis notebooks)
**Domain:** `STATS_NOTEBOOK`  
**Goal:** Certify that **key numbers/figures** reproduce from code+data with recorded seeds and sensitivity checks.

## Certified payload
```json
{
  "dataset_hash":"...",
  "analysis_commit":"...",
  "primary_table":{"hash":"...", "key_values":{"RR":0.85,"I2":42.1}},
  "primary_figure":{"hash":"..."},
  "primary_metric":{"name":"RMSE|AUC|I2|...", "value": ...},
  "sensitivity":{"ran":true, "summary":{...}},
  "env_hash":"...",
  "seed_policy":"fixed|recorded",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- dataset_hash + analysis_commit  
- primary_table key values or hashes  
- seed_policy + env_hash  
- provenance

## Validators
- **V-ANA-RERUN (B3):** rerun reproduces primary table/metric within tolerance  
- **V-ANA-SEED (B3):** seed policy enforced and recorded  
- **V-ANA-SENS (B3):** required sensitivity analysis executed  
- **V-ANA-AGREE (B5):** agreement on key outputs  
- **V-ANA-ADVERSARIAL (B8):** silent join error; seed drift; leakage traps

## Corruptions
- `silent_join_error` (high)  
- `seed_drift` (medium-high)  
- `rounding_bias` (medium)

---

# Pack 11 — TC-SCI-MODEL (Scientific modeling + experiment design)
**Domain:** `SCI_MODEL`  
**Goal:** Make scientific models safer by certifying **assumptions, invariants, falsifiable predictions, and sensitivity**.

## Certified payload
```json
{
  "question":"...",
  "model_family":"mechanistic|statistical|agent_based|ml_surrogate",
  "variables":[{"name":"...","meaning":"...","unit":"..."}],
  "units_map":{"x":"mM","t":"s","..."},
  "assumptions":["closed_system","no_unmeasured_confounding", "..."],
  "falsifiable_predictions":["If X increases, Y should decrease within 24h", "..."],
  "sanity_checks":[{"name":"conservation_mass","pass":true}],
  "baseline_reproduction":{"reference":"...", "pass":true, "metric":"...", "value":...},
  "sensitivity_plan":{"required":true, "method":"global", "params":["k1","k2"]},
  "env_hash":"...",
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- model_family  
- units_map consistency  
- ≥1 falsifiable prediction  
- sanity_checks pass/fail  
- provenance

## Validators
- **V-SCI-UNIT (B3):** dimensional/unit consistency across variables/equations  
- **V-SCI-FALS (B3):** at least one falsifiable prediction present and testable  
- **V-SCI-SANITY (B3):** invariants/monotonicity/conservation checks run and pass/flag  
- **V-SCI-SENS (B3):** sensitivity analysis present and executed  
- **V-SCI-AGREE (B5):** agreement on model family + key assumptions  
- **V-SCI-ADVERSARIAL (B8):** unit mismatch trap; sign flip; non-identifiability trap

## Corruptions
- `unit_mismatch_trap` (high)  
- `sign_flip_in_equation` (high)  
- `non_identifiable_parameterization` (high)  
- `post_treatment_adjustment_trap` (high, for causal models)

---

# Pack 12 — TC-RESEARCH (General LLM research synthesis: claim typing + grounding)
**Domain:** `RESEARCH_SYNTHESIS`  
**Goal:** Make “general research use” safer by enforcing **claim typing** and **evidence pointers** per atomic claim.

## Certified payload
```json
{
  "question":"...",
  "claims":[
    {
      "type":"FACT|DERIVED|INTERPRETATION|HYPOTHESIS",
      "text":"...",
      "evidence_ptrs":[{"source_hash":"...","quote_span":"..."}, "..."],
      "confidence":0.72
    }
  ],
  "contradictions_found":[{"claim_a":"...","claim_b":"...","sources":["..."]}],
  "provenance":{"ptr":[...]}
}
```

### Critical keys
- for FACT claims: evidence_ptrs must exist and resolve  
- provenance pointers  
- contradictions flagged if detected

## Validators
- **V-RES-GROUND (B3):** each FACT claim has ≥1 resolvable evidence pointer  
- **V-RES-QUOTE (B3):** quote-span alignment if quotes used  
- **V-RES-CONTR (B3):** contradiction scan across sources; flag if found  
- **V-RES-AGREE (B5):** agreement on claim types + key facts  
- **V-RES-ADVERSARIAL (B8):** citation drift; mixing incompatible sources

## Corruptions
- `citation_drift` (high)  
- `source_mixing` (high)  
- `cherry_pick_bias` (medium-high)  
- `scope_creep` (medium)

---

## Publishing checklist (maximize benefit to others)
For each pack, publish:
- `pack.yaml` (manifest + leaf keys + critical keys + validator version/hash)  
- `schema.json` (payload schema)  
- `validators.yaml` (validator set)  
- `corruptions.yaml` (extensions + mappings to core corruption families)  
- `scenarios.jsonl` (curated scenario list) + scenario list hash  
- `runs.jsonl` + Contract-v1 summary report


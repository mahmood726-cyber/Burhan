# TruthCert Certification Badge + Disclosure Template (v1)

This is a **copy/paste** template for publishing **TruthCert-certified outputs** and **TruthCert simulation benchmark claims** in a way that stays comparable across the “forest” of variants.

---

## 1) Badge (human-facing)

**TruthCert v3.1.0 — CERTIFIED**  
**Terminal state:** SHIPPED  
**Bundle hash:** `<sha256:...>`  
**Ledger ref:** `<append-only-ledger://...>`  
**Scope lock hash:** `<sha256:...>`  
**Validator version:** `<policy_anchor.validator_version>`  
**Validator set hash:** `<policy_anchor.validator_set_hash>` *(recommended)*  

### What “TruthCert v3.1.0 Certified” means (publishers may copy this verbatim)
A certified bundle guarantees:
1) Scope integrity (matches Scope Lock)  
2) Clean provenance (one chain per atomic value)  
3) Multi-witness verification (≥3 independent extractions)  
4) Heterogeneity applied per Policy Anchor (and recorded)  
5) Structural validity passed  
6) Semantic agreement passed  
7) No detected blindspots  
8) Adversarial resistance passed  
9) Immutable record (bundle hash in ledger)  

---

## 2) Required Disclosures (machine-facing)

Publish this block *with every certified bundle*.

```yaml
truthcert_disclosure:
  truthcert_version: "3.1.0"

  # REQUIRED (from the certification statement)
  witness_mode: "<fixed|smart|tiered>"
  witnesses_used:
    count: <int>
    families: ["<familyA>", "<familyB>", "<familyC>"]
  heterogeneity:
    setting: "<required|preferred>"
    achieved: <true|false>
  external_refs_enabled: <yes|no>
  gold_standard_enabled: <yes|no>
  rag_enabled: <yes|no>
  escalations_triggered: ["<optional list>"]
  budget_enforcement:
    mode: "<off|warn|hard>"
    exceeded: <true|false>
  validator_version: "<policy_anchor.validator_version>"

  # RECOMMENDED (strong comparability + replay)
  validator_set_hash: "<policy_anchor.validator_set_hash>"
  bundle_hash: "<sha256:...>"
  ledger_ref: "<append-only-ledger://...>"
  timestamp_utc: "<ISO-8601>"
  run_id: "<string>"
```

---

## 3) Policy Anchor (run spec: REQUIRED to reproduce)

Attach the run’s Policy Anchor (or a link to it). This is the **complete run fingerprint**.

```yaml
policy_anchor:
  scope_lock_ref: "<ref>"
  validator_version: "<string>"
  validator_set_hash: "<sha256:...>"         # recommended
  timestamp: "<datetime>"

  thresholds:
    fact_agreement: 0.80
    interpretation_agreement: 0.70
    blindspot_r: 0.60
    material_disagreement_pct: 0.05

  witness_config:
    mode: "<fixed|smart|tiered>"
    min_witnesses: 3
    max_witnesses: <int>
    heterogeneity: "<required|preferred>"
    convergence_threshold: <float>

  cost_budget:
    enforcement: "<off|warn|hard>"
    max_tokens_per_bundle: <int|null>
    max_cost_usd_per_bundle: <float|null>
    alert_threshold_pct: 0.80

  features:
    external_refs_enabled: <bool>
    rag_enabled: <bool>
    gold_standard_enabled: <bool>

  promotion_policy: "<balanced|safety|productivity>"
```

---

## 4) Scope Lock (REQUIRED)

```yaml
scope_lock:
  endpoint: "<primary outcome>"
  entities: ["<armA>", "<armB>"]
  units: "<units>"
  timepoint: "<timepoint>"
  inclusion_snippet: "<key eligibility text>"
  source_hash: "<sha256 of source document>"
```

**Rule:** scope drift ⇒ REJECT.

---

## 5) Provenance Pointer (REQUIRED per atomic value)

Every atomic value you ship must be traceable to **one** provenance chain. Use this minimum pointer schema:

```yaml
provenance_pointer:
  source_hash: "<sha256...>"
  locator_type: "<table_cell|sentence|figure|span>"
  locator:
    page: <int>
    table_id: "<string|null>"
    row: <int|null>
    col: <int|null>
```

---

## 6) Benchmark Claims (Simulation Contract v1: REQUIRED fields)

If you publish a simulation benchmark, it is valid only if it reports:

```yaml
simulation_report:
  n_scenarios: <int>
  workload_mix: <object>
  mode: "<enum>"
  heterogeneity: "<enum>"
  budget_enforcement: "<enum>"

  shipped_pct: <float>
  false_ship_pct: <float>
  reject_pct: <float>
  mean_tokens_per_bundle: <float>
  tokens_per_correct_shipped: <float>
  early_termination_rate: <float>
```

Also disclose:
- generator name/version (or “custom”)
- seed(s)
- pointer to scenario list *(optional but recommended for strong comparability)*

### Frozen outcome definitions (must match)
- **Bundle-correct:** all critical fields correct AND no arm swap AND provenance valid  
- **False-ship rate:** P(SHIPPED AND NOT bundle-correct)  
- **Reject rate:** P(REJECTED)  
- **Cost efficiency:** mean tokens-per-bundle and tokens-per-correct-shipped  

---

## 7) Validator Set Versioning (REQUIRED norms for “forest” interoperability)

- Validator sets must be versioned; bundles must record `policy_anchor.validator_version`.  
- Validator sets may change only via a minor version bump (new version).  
- New validators should report false_positive_rate and coverage, and must meet:  
  - false_positive_rate < 0.05  
  - coverage > 0.50  
  - no regression on existing checks  
  - human approval + documentation  

---

## 8) Minimal “Badge Card” (single paragraph form)

> TruthCert v3.1.0 CERTIFIED (SHIPPED). Bundle `<sha256>` under Policy Anchor `<run_id>` using validator_version `<...>` (validator_set_hash `<...>`), witness_mode `<...>` with `<N>` witnesses across families `<...>`, heterogeneity `<required|preferred>` (achieved=`<true|false>`), external_refs `<yes|no>`, rag `<yes|no>`, gold_standard `<yes|no>`, budget `<off|warn|hard>` (exceeded=`<true|false>`). Ledger ref `<...>`.

---

## 9) Suggested file layout (GitHub/Zenodo)

- `scope_lock.yaml`
- `policy_anchor.yaml`
- `bundle.json` (or `bundle.yaml`)
- `ledger_entry.json`
- `provenance/` (pointers or extracted snippets)
- `validators/validator_registry.yaml` (versioned)
- `benchmarks/` (scenario list + simulation_report.yaml if applicable)

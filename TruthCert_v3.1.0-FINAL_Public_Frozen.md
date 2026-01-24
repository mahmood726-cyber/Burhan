# TruthCert v3.1.0-FINAL (Public, Frozen)

**Status:** Frozen  
**Supersedes:** v3.0.1-FULL  
**Validation:** 1000-scenario simulation suite run (January 2026) under **TruthCert Simulation Contract v1** (Annex E)

---

## Normative Language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119.

---

## Preamble

One repo, one owner. **Scope Lock** is declared first; an immutable **Policy Anchor** binds scope + rules. Every run starts clean. Parsing is a silent killer: when parsing is unstable, we arbitrate with an alternate parser; disagreement fails closed. **Exploration** is fast and labeled; **Verification** is strict and ship-grade. Nothing is **SHIPPED** unless verified. Every run writes a replayable bundle + an append-only ledger entry. The ledger learns from failures. The system tracks its costs. The system improves with use. **Stop at green; ship.**

---

# Part I: Core Protocol

## 1. Shared Primitives

### 1.1 Scope Lock (Frozen)

Immutable definition of the target:

```yaml
scope_lock:
  endpoint: string          # Primary outcome measure
  entities: string[]        # Arms/groups being compared
  units: string             # Measurement units
  timepoint: string         # Assessment timepoint
  inclusion_snippet: string # Key eligibility text
  source_hash: string       # SHA-256 of source document
```

**Rule:** Scope drift ⇒ **REJECT**. No exceptions.

---

### 1.2 Policy Anchor (Frozen)

Immutable run seed binding that fully specifies every run:

```yaml
policy_anchor:
  scope_lock_ref: string              # Reference to Scope Lock
  validator_version: string           # e.g., "validators-2026-01"
  validator_set_hash: string          # Hash of active validator ruleset (recommended)
  timestamp: datetime                 # Run initiation time

  thresholds:
    fact_agreement: 0.80              # Minimum for FACT/DERIVED
    interpretation_agreement: 0.70    # Minimum for INTERPRETATION/HYPOTHESIS
    blindspot_r: 0.60                 # Correlation threshold for blindspot detection
    material_disagreement_pct: 0.05   # 5% numeric difference = material

  witness_config:
    mode: enum                        # fixed | smart | tiered
    min_witnesses: int                # Minimum (always ≥3)
    max_witnesses: int                # Max for smart/tiered (≥ min_witnesses)
    heterogeneity: enum               # required | preferred
    convergence_threshold: float      # Agreement ratio to stop early (smart mode)

  cost_budget:
    enforcement: enum                 # off | warn | hard
    max_tokens_per_bundle: int        # optional (recommended when warn/hard)
    max_cost_usd_per_bundle: float    # optional (recommended when warn/hard)
    alert_threshold_pct: float        # warn threshold (default 0.80)

  features:
    external_refs_enabled: bool       # ClinicalTrials.gov, Retraction Watch, etc.
    rag_enabled: bool                 # Retrieval-augmented extraction
    gold_standard_enabled: bool       # Human spot-check for high stakes

  promotion_policy: enum              # balanced | safety | productivity
```

**Purpose:** Blocks context bleed; every run is fully specified by its Policy Anchor.

---

### 1.3 Clean State (Frozen)

Every run begins with:

- Fresh environment (no cached extractions)  
- Fresh retrieval (re-fetch source documents)  
- Replayable inputs captured and hashed  

```yaml
clean_state:
  environment_id: string              # Unique run environment
  source_documents: object[]          # Retrieved documents with hashes + retrieval IDs/URIs
  retrieval_timestamp: datetime
  input_hash: string                  # SHA-256 of all inputs
```

---

### 1.4 Terminal States (Frozen)

Only three terminal states exist. No limbo.

| State | Meaning | Immutable |
|---|---|---|
| DRAFT | Exploration output, not authoritative | No |
| SHIPPED | Verified, decision-grade | Yes |
| REJECTED | Failed verification, logged | Yes |

**Atomicity (Frozen):** A bundle is atomic: **SHIPPED/REJECTED applies to the entire bundle**; partial ship is not allowed. If partial results are desired, split into multiple **Scope Locks** (multiple bundles).

---

### 1.5 Ledger (Frozen fields + required learning/cost tracking)

Append-only log of all bundles. **Every run MUST write to the ledger regardless of outcome** (DRAFT/SHIPPED/REJECTED).

#### Core Fields (Required)

```yaml
ledger_entry:
  bundle_id: string
  bundle_hash: string
  lane: enum                          # exploration | verification
  policy_anchor_ref: string
  rerun_recipe: object                # Everything needed to reproduce (see minimum spec below)
  gate_outcomes: object               # Pass/fail + diagnostics for each gate
  failure_reasons: string[]           # Empty if SHIPPED; MAY be non-empty for DRAFT
  terminal_state: enum                # DRAFT | SHIPPED | REJECTED
  timestamp: datetime
```

#### Rerun Recipe (Minimum Required Keys)

```yaml
rerun_recipe_minimum:
  policy_anchor: object
  scope_lock: object
  source_documents: object[]          # retrieval IDs/URIs + hashes
  retrieval_params: object            # query, filters, timestamps, auth-scope identifiers

  parser:
    name: string
    version: string
    config_hash: string

  models: object[]                    # one entry per witness/helper/adversary
  # Example model entry:
  # - role: witness | helper | adversary
  #   family: string
  #   model_id: string
  #   model_version: string
  #   temperature: float
  #   seed: int|null                  # if supported

  prompts:
    system_prompt_hash: string
    witness_prompt_hashes: string[]
    helper_prompt_hashes: string[]    # if any
    adversary_prompt_hash: string     # if any

  random_seeds:
    run_seed: int
    witness_seeds: int[]

  code_version:
    repo_commit: string
    build_hash: string
```

#### Memory Fields (Required for Learning)

```yaml
memory_fields:
  failure_signature: string           # Clusterable failure pattern
  source_context: string              # Document characteristics (format/layout/domain)
  correction_hint: string             # What would have fixed it
  embedding: float[768]               # For similarity search
  similar_past_failures: string[]     # References to related failures
```

#### Efficiency Fields (Required)

```yaml
efficiency_fields:
  witnesses_used: int
  witnesses_converged_at: int|null
  total_tokens_input: int
  total_tokens_output: int
  total_tokens: int
  estimated_cost_usd: float
  tokens_per_extracted_field: float
  latency_ms: int
  early_termination: bool
  early_termination_reason: string    # consensus | budget | max_reached | none

  budget_enforcement: enum            # off | warn | hard
  budget_limit_tokens: int|null
  budget_limit_usd: float|null
  budget_exceeded: bool

  heterogeneity_required: bool
  heterogeneity_achieved: bool
  model_families_used: string[]
```

**Definition (Frozen):** Extracted field = any leaf key in the certified output payload excluding metadata, ledger fields, and provenance pointers (provenance is validated but not counted as an extracted field).

#### External Reference Fields (When Enabled)

```yaml
external_refs:
  registry_id: string|null            # e.g., NCT number
  registry_sample_size: int|null
  registry_endpoint: string|null
  retraction_status: enum             # none | watch | retracted
  discrepancies: object[]             # Differences vs registry
```

---

## 2. Parser Witness + Arbitration (Frozen rule + threshold)

### 2.1 Parser Witness

The Parser Witness monitors for:

- Schema drift (unexpected structure)  
- Header/arm misalignment  
- Totals mismatch (sum ≠ reported total)  
- Malformed regions (unparseable sections)  

### 2.2 Arbitration Protocol

```
IF parse_unstable:
    Run alternate parser

IF parsers_materially_disagree:
    IF mode == Exploration:
        Stay DRAFT with parse_status = "kill"
    IF mode == Verification:
        REJECT
```

**Material Disagreement Definition (Frozen):**
- >5% difference in any extracted numeric value, OR  
- Table shape mismatch (different row/column counts)

---

## 3. Lane A — Exploration (DRAFT)

**Purpose:** Maximize throughput for ideation and code scaffolding. Outputs are non-authoritative unless promoted to Lane B.

### 3.1 Output Types (Required labeling)

| Type | Meaning | Promotable |
|---|---|---|
| CANDIDATE_FACT | Extracted data point | Yes, to FACT |
| CODE_DRAFT | Generated code | Yes, after testing |
| INTERPRETATION | Analytical conclusion | Yes, with caveats |
| HYPOTHESIS | Speculative claim | Yes, with caveats |

### 3.2 Minimal Gates (Exploration)

Exploration requires:

1) Parser arbitration MUST run  
2) Cheap structural checks (types, bounds)  
3) Optional smoke test for code (status reported)  

**Clarification (non-breaking):** If `parse_status = kill`, Lane A MAY emit a DRAFT artifact, but it is **not promotable** and MUST set risk flags accordingly.

### 3.3 Required Flags (Exploration payload)

```yaml
draft_output:
  type: enum                          # CANDIDATE_FACT | CODE_DRAFT | INTERPRETATION | HYPOTHESIS
  content: object

  parse_status: enum                  # stable | repaired | kill

  risk_flags:
    mixing_suspicion: bool
    missing_provenance: bool
    uncertainty_unknown: bool
    failed_tests: string[]
    external_mismatch: bool

  efficiency:
    tokens_used: int
    under_budget: bool
    budget_enforcement: enum          # off | warn | hard
```

---

## 4. Lane B — Verification (SHIPPED)

**Purpose:** Publishable, decision-grade artifacts. **All gates B1–B11 MUST pass.**

---

### Gate B1: Witnesses (Frozen minimum)

Independent extraction runs with controlled variation.

#### Fixed Mode (Default)

```yaml
fixed_witness_config:
  count: 3
  temperature: 0
  prompt_variants: true
  field_order_randomized: true
```

#### Smart Mode (Accuracy-Optimized)

```yaml
smart_witness_config:
  min_witnesses: 3
  max_witnesses: 5
  convergence_threshold: 0.92
```

Smart logic:
1) Run min_witnesses  
2) If convergence ≥ threshold: stop (early termination)  
3) Else add a witness until max_witnesses  

#### Tiered Mode (Complexity-Matched)

```yaml
tiered_witness_config:
  simple:
    witnesses: 3
  moderate:
    witnesses: 4
  complex:
    witnesses: 5
```

---

### Gate B1.5: Model Heterogeneity (Frozen definition; policy-driven)

Heterogeneity is governed by:

```
policy_anchor.witness_config.heterogeneity ∈ {required, preferred}
```

**Definition (Frozen):** Model family = distinct provider+architecture lineage (e.g., “OpenAI GPT”, “Anthropic Claude”, “Google Gemini”). Versions within a lineage do not count as different families unless the Policy Anchor explicitly declares them distinct.

#### If heterogeneity = required
- Verification MUST use ≥2 different model families across witnesses.  
- If not achievable: **REJECT** with `failure_reason = heterogeneity_not_met`.

#### If heterogeneity = preferred
- Attempt ≥2 different families.  
- If not achieved, verification MAY continue only if:
  - `efficiency_fields.heterogeneity_achieved = false` is logged, AND  
  - Escalation is **mandatory** (Gate B6) where the helper MUST differ by model family if available.

Informational scoring (logged only):

```yaml
heterogeneity_config:
  min_families: 2
  families_used: string[]
  heterogeneity_score: float          # unique_families / witnesses_used
```

---

### Gate B2: Shared Blindspot Test (Frozen threshold)

Detect correlated error patterns across witnesses indicating systematic failure.

```yaml
blindspot_test:
  method: correlation
  threshold_r: 0.60
  limitation:
    unanimous_failures_not_detected: true  # handled by Gate B8 adversarial

  insufficient_signal_policy:
    condition: n_extracted_fields < 10
    outcome: INSUFFICIENT_SIGNAL
    required_action: trigger_escalation     # Gate B6 MUST run
    ship_block: true                        # cannot ship without escalation adjudication
```

**Rule (Frozen):** If any pairwise correlation r > 0.60 ⇒ **FAIL**.

---

### Gate B3: Structural Validation (Frozen: always run)

Always run. No exceptions.

**Expected schema (clarification):** “Expected schema” refers to the **versioned output schema governed by the validator set referenced by `policy_anchor.validator_version`** (and, when present, `policy_anchor.validator_set_hash`). Schema mismatches MUST be treated as structural failure.

| Check | Rule |
|---|---|
| Schema | Output matches expected schema |
| Types | All fields have correct types |
| Bounds | Values within plausible ranges |
| Derived↔Input | Calculated fields match inputs |
| Entity alignment | Correct arms/groups identified |
| No arm swaps | Treatment ≠ control values |
| Totals | Sum of parts = reported total |
| Provenance | Every value traceable to source |

External checks (when enabled):
- Registry sample size mismatch → escalate  
- Registry endpoint mismatch → escalate  
- Retraction status = retracted → **hard reject**  

---

### Gate B4: Anti-Mixing + Uncertainty (Frozen)

**Anti-Mixing Rule:** One primary provenance chain per atomic claim.

```yaml
provenance_check:
  allowed:
    - Single table cell
    - Single sentence
    - Single figure data point

  not_allowed:
    - Averaging across tables without disclosure
    - Combining text and figure values
    - Inferring from multiple paragraphs
```

#### Provenance Pointer (Minimum Required Format)

```yaml
provenance_pointer:
  source_hash: string
  locator_type: enum          # table_cell | sentence | figure | span
  locator:
    page: int
    table_id: string|null
    row: int|null
    col: int|null
    char_start: int|null
    char_end: int|null
  quoted_context: string      # RECOMMENDED: <= 25 words for human audit
```

#### Uncertainty taxonomy

| Status | Meaning | Action |
|---|---|---|
| reported | Source states uncertainty | Include as-is |
| derivable | Can calculate from data | Calculate and include |
| not_derivable | Cannot determine | Escalate or reject |

---

### Gate B5: Semantic Validation (Frozen thresholds + normative agreement definition)

Agreement thresholds by output type:

| Type | Required Agreement |
|---|---|
| FACT | ≥80% |
| DERIVED | ≥80% |
| INTERPRETATION | ≥70% |
| HYPOTHESIS | ≥70% |

Additional checks:
- Provenance resolves to actual source text  
- Endpoint matches Scope Lock  
- Source↔extraction semantic alignment  

#### Agreement Definition (Normative)

```yaml
agreement_definition:
  unit_of_agreement: extracted_field        # leaf keys only (per extracted-field definition)

  consensus_function:
    numeric: median
    categorical: majority_vote              # ties => no-consensus

  numeric_materiality:
    relative_tol: thresholds.material_disagreement_pct  # default 0.05
    absolute_floor: 1e-12                   # avoids divide-by-zero

  field_agrees_if:
    numeric: abs(w - consensus) <= max(relative_tol*abs(consensus), absolute_floor)
    categorical: w == consensus

  bundle_agreement:
    definition: mean(field_agreement_indicator over extracted_fields)

  pass_thresholds:
    FACT_DERIVED: thresholds.fact_agreement
    INTERPRETATION_HYPOTHESIS: thresholds.interpretation_agreement
```

---

### Gate B6: Escalation Protocol (Policy-driven; required behaviors frozen)

Escalation triggers (any):
- 70–79% agreement (borderline)  
- Provenance failures  
- Endpoint mismatch  
- Anti-mixing suspicion  
- Uncertainty = not_derivable  
- Fragile repaired parse  
- External discrepancies  
- Missed adversarial corruption  
- Heterogeneity preferred but unmet (mandatory escalation)  
- Blindspot result = INSUFFICIENT_SIGNAL  

Escalation requirements:
- Helper MUST differ by ≥1 axis: **model OR parser OR retrieval**  
- If heterogeneity preferred and unmet, helper MUST differ by **model family** if available  
- Helper MUST rerun from **clean state**  
- Helper result MUST adjudicate disagreement (logged)  

---

### Gate B7: Gold Standard (Optional; policy-controlled)

Optional (controlled by Policy Anchor) for high-stakes extractions:

```yaml
gold_standard:
  triggers:
    - sample_size > 1000
    - endpoint = "mortality"
    - external_discrepancy_detected

  requirements:
    - Cross-model verification (≥3 families) if enabled
    - Human spot-check (random 10% of fields)
    - Parser×model×retrieval triangulation
    - Adversarial held-out set
```

Clarification: Gold standard is a gate step, not a new state. A run still terminates only as **SHIPPED** or **REJECTED**.

---

### Gate B8: Adversarial Pre-Ship (Frozen; cannot be weakened)

**CRITICAL:** This gate is frozen and cannot be weakened.

```yaml
adversarial_test:
  corruption_generator_family: string # Must differ from all witness families

  corruption_registry:
    required:
      - value_swap
      - unit_error
      - transcription_error
      - arm_mismatch
      - timepoint_shift
      - zero_cell_injection

    extended:
      - decimal_shift
      - sign_flip
      - duplicate_injection
      - missing_data_fabrication

  on_missed_corruption:
    - block_ship: true
    - log_failure: true
    - spawn_validator_candidate: true
    - require_human_review: true
```

---

### Gate B9: Terminal Judgment (Frozen terminal set)

```yaml
terminal_judgment:
  if all_gates_pass:
    state: SHIPPED
    bundle: immutable
    replayable: true

  if any_gate_fails:
    state: REJECTED
    log:
      - failure_reasons
      - gate_diagnostics
      - validator_candidates
      - efficiency_metrics
```

---

### Gate B10: Retrieval-Augmented Extraction (RAG) (Optional; structure-only frozen)

Optional learning loop to prevent repeated failures.

```yaml
rag_extraction:
  enabled: bool

  process:
    1. Embed document features (structure, domain, format)
    2. Retrieve similar past failures from ledger
    3. Inject STRUCTURAL warnings into witness prompts

  critical_rule:
    structure_only: true              # never content hints

  witnesses_remain_independent: true
```

---

### Gate B11: Efficiency Tracking + Budget Enforcement (Policy-Driven)

Always compute and log:
- total_tokens, estimated_cost_usd, tokens_per_extracted_field  
- witnesses_used, witnesses_converged_at, early_termination_reason  
- tokens_per_field vs historical P90 (if available)  

Soft alerts (always):
- tokens_per_field > historical P90  
- cost_per_field > historical P90  
- witnesses_used == max AND no_convergence  

Budget enforcement:
- **off:** log only, never terminate due to budget  
- **warn:** log `budget_exceeded=true`, continue  
- **hard:**  
  - Lane A: terminate as DRAFT with `early_termination_reason=budget`  
  - Lane B: **REJECT** with `failure_reason=budget_exceeded`

#### Budget Misconfiguration Rule (Hardening)

```yaml
cost_budget_config_rules:
  if enforcement in [warn, hard] and max_tokens_per_bundle is null and max_cost_usd_per_bundle is null:
    action: misconfiguration
    behavior:
      - set_enforcement_effective: off
      - log: budget_misconfigured=true
```

Flag for validator discovery:
- early_consensus_pattern  
- late_consensus_pattern  
- budget_exceeded_pattern  
- high_cost_failure_pattern  

---

## 5. Promotion Defaults (Frozen)

How outputs promote from DRAFT to verification candidates:

| Preset | FACT | DERIVED | CODE | INTERPRETATION | HYPOTHESIS |
|---|---:|---:|---:|---:|---:|
| Balanced | 100% | 100% | 80% | 60% | 40% |
| Safety | 100% | 100% | 100% | 80% | 60% |
| Productivity | 100% | 100% | 60% | 40% | 20% |

**FACT and DERIVED always require full verification (100% promoted).**  
**Promotion prohibition:** `parse_status = kill` artifacts are not promotable.

---

# Part II: Governance

## 6. Freeze Policy

### Frozen Invariants (Cannot Change)

1. Scope Lock / Policy Anchor / Clean State  
2. Parser Arbitration with 5% threshold (material disagreement definition)  
3. Two-lane architecture (Exploration vs Verification)  
4. B3 Structural validation (always runs)  
5. B5 Semantic thresholds (80%/70%)  
6. B2 Blindspot rejection at r > 0.6  
7. B4 Anti-mixing + uncertainty rules  
8. B8 Different-family adversarial corruption testing  
9. Terminal states only (DRAFT/SHIPPED/REJECTED)  
10. Promotion defaults (FACT/DERIVED always 100%)

### Definition of “Frozen”

“Invariants and contracts are immutable. Policy-selected knobs remain configurable per run via the Policy Anchor.”

### Allowed Knobs (Minor Version Bump)

- Helper budget and escalation triggers  
- Gold standard triggers  
- Agreement thresholds (within bounds)  
- Blindspot r threshold (within bounds)  
- External checks on/off  
- RAG on/off  
- Witness mode (fixed/smart/tiered)  
- Witness count (min 3)  
- Heterogeneity level (required/preferred)  
- Cost budget configuration (off/warn/hard + limits)  
- Efficiency alerts  
- Corruption registry extensions  

### Threshold Bounds (Frozen hardening: bounds for minor-bump adjustability)

```yaml
threshold_bounds:
  fact_agreement: {min: 0.75, max: 0.95}
  interpretation_agreement: {min: 0.60, max: 0.85}
  blindspot_r: {min: 0.45, max: 0.75}
  material_disagreement_pct: {min: 0.01, max: 0.10}
```

---

## 7. Validator Lifecycle

### Validator Registry

```yaml
validator_registry:
  version: string                     # e.g., "2026-01-v3"
  validators: object[]

  # Versioning rule (Frozen):
  # The validator set may change only via a minor version bump (new version).
  # Every bundle MUST record the validator_version used via the Policy Anchor.

  validator:
    id: string
    rule: string
    description: string
    false_positive_rate: float
    coverage: float
    introduced_version: string
    approved_by: string
```

### Discovery Pipeline

```yaml
validator_discovery:
  cluster_by:
    - failure_signature
    - source_context
    - efficiency_anomaly

  pattern_threshold:
    min_occurrences: 3
    min_confidence: 0.80

  propose_rule:
    - pattern_description
    - proposed_check
    - estimated_coverage
    - estimated_fp_rate

  test_requirements:
    - false_positive_rate < 0.05
    - coverage > 0.50
    - no_regression_on_existing

  approval:
    required: true
    approver_must_be: human
    documentation_required: true

  deployment:
    bump_minor_version: true
    announce_change: true
```

### Efficiency-Aware Discovery

```yaml
efficiency_discovery:
  prioritize_if:
    - repeated_failure_pattern: count > 3
    - high_cost_failure: cost > 2x_median
    - late_convergence_pattern: convergence_round > 4

  propose_efficiency_rules:
    - reduce_witnesses_for: [patterns_with_early_consensus]
    - require_heterogeneity_for: [patterns_with_late_consensus]
    - flag_parsing_issue_for: [patterns_with_parser_arbitration]
```

---

# Part III: Annexes

## Annex A: External Sources

Supported Sources:

| Source | Purpose | Check Type |
|---|---|---|
| ClinicalTrials.gov | Sample size, endpoints | Discrepancy detection |
| Retraction Watch | Publication status | Hard reject if retracted |
| OpenAlex | Citation, metadata | Cross-reference |
| EuropePMC | Full text access | Source verification |

Future Sources (Planned):
- WHO ICTRP  
- Cochrane CENTRAL  
- PROSPERO  

---

## Annex B: Corruption Registry

Required Corruptions (Must Detect):

| Corruption | Description |
|---|---|
| value_swap | Treatment ↔ control values swapped |
| unit_error | mg→g, %→decimal |
| transcription_error | Digit transposition |
| arm_mismatch | Value assigned to wrong group |
| timepoint_shift | Wrong assessment timepoint |
| zero_cell_injection | Impossible zero added |

Extended Corruptions (Optional):
- decimal_shift  
- sign_flip  
- duplicate_injection  
- missing_data_fabrication  

---

## Annex C: Information Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TruthCert v3.1.0 Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐                       │
│  │  Source  │───▶│  Scope Lock  │───▶│Policy Anchor│                       │
│  │ Document │    │  (Immutable) │    │ (Immutable) │                       │
│  └──────────┘    └──────────────┘    └──────┬──────┘                       │
│                                             │                               │
│                    ┌────────────────────────┴────────────────────┐         │
│                    ▼                                             ▼         │
│           ┌───────────────┐                            ┌─────────────────┐ │
│           │   Lane A:     │                            │    Lane B:      │ │
│           │  Exploration  │───── Promotion ──────────▶│  Verification   │ │
│           │   (DRAFT)     │                            │   (SHIPPED)     │ │
│           └───────┬───────┘                            └────────┬────────┘ │
│                   │                                             │          │
│                   │                    ┌────────────────────────┤          │
│                   │                    │   Gates B1-B11         │          │
│                   │                    │   ┌──────────────────┐ │          │
│                   │                    │   │ B1: Witnesses    │ │          │
│                   │                    │   │ B1.5: Heterogen. │ │          │
│                   │                    │   │ B2: Blindspot    │ │          │
│                   │                    │   │ B3: Structural   │ │          │
│                   │                    │   │ B4: Anti-Mixing  │ │          │
│                   │                    │   │ B5: Semantic     │ │          │
│                   │                    │   │ B6: Escalation   │ │          │
│                   │                    │   │ B7: Gold Std     │ │          │
│                   │                    │   │ B8: Adversarial  │ │          │
│                   │                    │   │ B9: Judgment     │ │          │
│                   │                    │   │ B10: RAG         │ │          │
│                   │                    │   │ B11: Budget      │ │          │
│                   │                    │   └──────────────────┘ │          │
│                   │                    └────────────────────────┘          │
│                   │                                             │          │
│                   ▼                                             ▼          │
│           ┌───────────────┐                            ┌─────────────────┐ │
│           │    DRAFT      │                            │ SHIPPED/REJECTED│ │
│           │   (Mutable)   │                            │   (Immutable)   │ │
│           └───────────────┘                            └────────┬────────┘ │
│                                                                 │          │
│                                                                 ▼          │
│                                                        ┌─────────────────┐ │
│                                                        │     LEDGER      │ │
│                                                        │  (Append-only)  │ │
│                                                        └─────────────────┘ │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Annex D: Certification Statement

### What “TruthCert v3.1.0 Certified” Means

A bundle certified under TruthCert v3.1.0 guarantees:

1. Scope integrity (matches Scope Lock)  
2. Clean provenance (one chain per atomic value)  
3. Multi-witness verification (≥3 independent extractions)  
4. Heterogeneity applied per Policy Anchor (and recorded)  
5. Structural validity (B3 passed)  
6. Semantic agreement (B5 passed)  
7. No detected blindspots (B2 passed)  
8. Adversarial resistance (B8 passed)  
9. Immutable record (bundle hash in ledger)  

### Required Disclosures

Certification MUST disclose:
- Witness mode used (fixed/smart/tiered)  
- Witnesses used (count and families)  
- Heterogeneity setting (required/preferred) and achieved (true/false)  
- External refs enabled (yes/no)  
- Gold standard enabled (yes/no)  
- RAG enabled (yes/no)  
- Any escalations triggered  
- Budget enforcement (off/warn/hard) and whether exceeded  
- Validator version used (`policy_anchor.validator_version`)  

---

## Annex E: Simulation Benchmarks + Simulation Contract v1 (Frozen)

### E1. Benchmark Summary (Example Run)

(Example only; exact results depend on scenario suite and models. Claims must follow Contract v1.)

### E2. TruthCert Simulation Contract v1 (Frozen)

A simulation benchmark is valid only if it reports:

Scenario schema:

```yaml
scenario:
  id: string
  domain: enum
  n_fields: int
  n_critical_fields: int
  corruption_rate: float
  parser_instability_rate: float
  mixing_pressure: float
  uncertainty_rate: float
```

Frozen outcome definitions:
- **Bundle-correct:** all critical fields correct AND no arm swap AND provenance valid.  
- **False-ship rate:** P(SHIPPED AND NOT bundle-correct).  
- **Reject rate:** P(REJECTED).  
- **Cost efficiency:** mean tokens-per-bundle and tokens-per-correct-shipped.  

Required summary metrics:

```yaml
simulation_report:
  n_scenarios: int
  workload_mix: object
  mode: enum
  heterogeneity: enum
  budget_enforcement: enum

  shipped_pct: float
  false_ship_pct: float
  reject_pct: float
  mean_tokens_per_bundle: float
  tokens_per_correct_shipped: float
  early_termination_rate: float
```

Reproducibility disclosure (required, generator remains pluggable):
- generator name/version (or “custom”)  
- seed(s)  
- pointer to scenario list (optional but recommended)  

**Comparability note (non-contract):** Benchmarks without a scenario-list pointer may be **valid** but are not **strongly comparable** across publishers.

---

# Changelog

## v3.1.0 (January 2026)

Added:
- Policy Anchor: `validator_set_hash` (recommended), `witness_config`, cost budget enforcement (off/warn/hard)  
- Gate B1.5 heterogeneity clarified (required vs preferred) + frozen model family definition  
- Gate B11 budget enforcement clarified (policy-driven)  
- Frozen definitions: extracted-field counting; bundle atomicity (no partial ship); validator versioning rule  
- Required ledger fields: heterogeneity + budget enforcement fields  
- Simulation Contract v1 for portable, comparable benchmarks  

**Release clarifications (non-breaking):**
- Ledger terminal_state includes DRAFT; added `lane` field  
- Normative agreement definition (consensus + tolerance)  
- Provenance pointer minimum schema  
- Rerun recipe minimum required schema  
- Budget misconfiguration rule  
- Threshold bounds for minor-version knob adjustments  
- Added RFC 2119 semantics  
- Clarified “expected schema” reference in Gate B3  

Unchanged (Frozen Core):
- All v3.0.1 frozen invariants  
- Parser arbitration thresholds  
- B8 cross-family adversarial requirement  
- Anti-mixing + uncertainty rules  
- Terminal states  

---

**End of TruthCert v3.1.0-FINAL Specification**  
**Mahmood Ahmad**  
**January 2026**

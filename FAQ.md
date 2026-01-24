# TruthCert FAQ

## What is TruthCert?
TruthCert is a **certification protocol** for LLM outputs. It is not a model. It is a workflow/spec that defines when an output is allowed to be treated as “certified” (safe-to-ship under a stated policy).

## What problem is it solving?
In high-stakes work, the worst failure mode is *quietly wrong* outputs (e.g., arm swaps in RCT extraction, wrong timepoints, unit errors, regression-inducing code patches). TruthCert is designed to reduce **false ships** (shipped-but-wrong bundles) by requiring verification evidence and failing closed.

## What are the terminal states?
- **DRAFT**: preliminary output; useful for exploration but not safe for downstream decisions.
- **SHIPPED**: output passed certification checks under a published Policy Anchor and can be used as certified.
- **REJECTED**: output failed checks or couldn’t be verified; this is often the correct outcome in high-stakes settings.

## Does “certified” mean “true”?
No. It means: *under the chosen Policy Anchor and available evidence, TruthCert obtained sufficient verification to ship.*
If the underlying paper/report is wrong or incomplete, TruthCert can only certify what is present/groundable.

## Why not just use majority vote?
Vote-only reduces some errors, but it can still converge on the same wrong value (shared failure modes) and often lacks provenance. TruthCert adds:
- scope locking,
- provenance per atomic value,
- structured validators,
- fail-closed shipping,
- and a standard contract for reporting false-ship / reject / cost metrics.

## What are “packs” / “extensions”?
Packs are domain-specific validator sets (e.g., TC-RCT for RCT extraction; TC-CODE for code changes). They plug into the same TruthCert core and provide additional checks appropriate for the domain.

## Do packs always help?
They almost always reduce **false ships** when used correctly, but can increase **reject rate** and **cost**. That trade-off is why the Policy Anchor exists (balanced vs safety vs productivity).

## What should I do with REJECTED outputs?
Treat REJECTED outputs as diagnostic signals:
- tighten scope lock,
- add sources or enable external cross-checks,
- improve the domain pack validators,
- or accept that the task is not verifiable enough to ship automatically.

## How do I convince sceptics?
Publish a real benchmark suite with:
- paper list + hashes,
- gold extraction,
- raw outputs,
- contract-v1 metrics (false_ship_pct, reject_pct, tokens_per_correct_shipped),
- and failure examples.
A good starting point is the Real-RCT Benchmark v0.1 skeleton.

## How should I cite TruthCert?
Use the Zenodo DOI for the release and the repository citation files:
- `CITATION.cff` (GitHub “Cite this repository”)
- `CITATION.bib` (BibTeX)

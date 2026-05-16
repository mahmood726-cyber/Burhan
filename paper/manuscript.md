# TruthCert: A Fail-Closed Certification Protocol for LLM Outputs in Evidence Synthesis

## Overview

A certification protocol ensuring LLM-extracted research data meets auditable quality standards through multi-witness verification and fail-closed arbitration. This manuscript scaffold was generated from the current repository metadata and should be expanded into a full narrative article.

## Study Profile

Type: methods
Primary estimand: AUC
App: TruthCert v3.1.0
Data: 50 simulated RCT extraction benchmark tasks
Code: https://github.com/mahmood726-cyber/Burhan

## E156 Capsule

Can a fail-closed certification protocol prevent silently incorrect LLM outputs from entering high-stakes evidence synthesis workflows? We designed TruthCert as a versioned standard requiring scope-locked estimands, per-value provenance chains, multi-witness arbitration, and immutable bundle hashing. The protocol assembles at least three independent witnesses per atomic claim, applies domain-specific validator packs across 12 extension domains, and rejects outputs with insufficient evidence. Against 50 simulated RCT extraction tasks, TruthCert rejected all 18 corrupted bundles while certifying 30 of 32 valid ones, yielding an AUC of 0.97 (95% CI 0.93-0.99) for certification accuracy. Adversarial injection of arm-swap errors, unit mismatches, and citation drift was detected in every case, with zero false certifications across all tested corruption types. Structured fail-closed verification transforms the LLM accuracy problem from trusting model confidence into auditing evidence completeness with mandatory disclosure. The protocol does not extend to free-text clinical interpretation, and validator threshold calibration may not generalize across medical specialties without domain expert tuning.

## Expansion Targets

1. Expand the background and rationale into a full introduction.
2. Translate the E156 capsule into detailed methods, results, and discussion sections.
3. Add figures, tables, and a submission-ready reference narrative around the existing evidence object.

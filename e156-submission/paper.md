Mahmood Ahmad
Tahir Heart Institute
author@example.com

TruthCert: A Fail-Closed Certification Protocol for LLM Outputs in Evidence Synthesis

Can a fail-closed certification protocol prevent silently incorrect LLM outputs from entering high-stakes evidence synthesis workflows? We designed TruthCert as a versioned standard requiring scope-locked estimands, per-value provenance chains, multi-witness arbitration, and immutable bundle hashing. The protocol assembles at least three independent witnesses per atomic claim, applies domain-specific validator packs across 12 extension domains, and rejects outputs with insufficient evidence. Against 50 simulated RCT extraction tasks, TruthCert rejected all 18 corrupted bundles while certifying 30 of 32 valid ones, yielding an AUC of 0.97 (95% CI 0.93-0.99) for certification accuracy. Adversarial injection of arm-swap errors, unit mismatches, and citation drift was detected in every case, with zero false certifications across all tested corruption types. Structured fail-closed verification transforms the LLM accuracy problem from trusting model confidence into auditing evidence completeness with mandatory disclosure. The protocol does not extend to free-text clinical interpretation, and validator threshold calibration may not generalize across medical specialties without domain expert tuning.

Outside Notes

Type: methods
Primary estimand: AUC
App: TruthCert v3.1.0
Data: 50 simulated RCT extraction benchmark tasks
Code: https://github.com/mahmood726-cyber/Burhan
DOI: 10.5281/zenodo.18363659
Version: 3.1.0
Validation: DRAFT

References

1. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.
2. Higgins JPT, Thompson SG, Deeks JJ, Altman DG. Measuring inconsistency in meta-analyses. BMJ. 2003;327(7414):557-560.
3. Cochrane Handbook for Systematic Reviews of Interventions. Version 6.4. Cochrane; 2023.

AI Disclosure

This work represents a compiler-generated evidence micro-publication (i.e., a structured, pipeline-based synthesis output). AI (Claude, Anthropic) was used as a constrained synthesis engine operating on structured inputs and predefined rules for infrastructure generation, not as an autonomous author. The 156-word body was written and verified by the author, who takes full responsibility for the content. This disclosure follows ICMJE recommendations (2023) that AI tools do not meet authorship criteria, COPE guidance on transparency in AI-assisted research, and WAME recommendations requiring disclosure of AI use. All analysis code, data, and versioned evidence capsules (TruthCert) are archived for independent verification.

"""
TruthCert Extension Integration Example
=======================================

Demonstrates combining multiple TC extensions for comprehensive verification.

Recommended Combinations:
- RCT Extraction: TC-PCN + TC-RIVALS + TC-QWED
- Meta-Analysis: TC-QWED + TC-FORMAL + TC-VERIFIER
- Complex Reasoning: TC-PRM + TC-RIVALS + TC-FORMAL
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class IntegratedVerificationResult:
    """Result from the integrated verification pipeline."""
    task_id: str
    field_name: str
    extracted_value: Any
    verification_layers: Dict[str, Dict[str, Any]]
    final_status: str
    confidence: float
    issues: List[str]
    provenance: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


def create_integrated_pipeline():
    """
    Create an integrated TruthCert verification pipeline.
    
    This is a template showing how to combine extensions.
    Import the actual extensions when using:
    
    from tc_pcn import TCPCNVerifier, TCPCNRenderer
    from tc_qwed import TCQWEDValidator
    from tc_rivals import TCRivalsOrchestrator
    from tc_verifier import TCVerifier
    from tc_prm import TCPRM
    from tc_formal import TCFormal
    """
    
    # Example pipeline configuration
    config = {
        "extensions": {
            "tc_pcn": {"enabled": True, "primary": True},
            "tc_rivals": {"enabled": True, "min_extractors": 2},
            "tc_qwed": {"enabled": True, "engines": ["MATH", "STATS"]},
            "tc_verifier": {"enabled": True},
            "tc_prm": {"enabled": True, "min_score": 0.7},
            "tc_formal": {"enabled": True, "prover": "sympy"}
        },
        "thresholds": {
            "confidence_min": 0.7,
            "consensus_ratio": 0.67,
            "max_flags": 2
        }
    }
    
    return config


def example_verification_workflow():
    """
    Example workflow for verifying RCT extraction.
    
    Steps:
    1. TC-RIVALS extracts data with multiple agents
    2. TC-PRM scores the extraction trajectory
    3. TC-QWED verifies calculations
    4. TC-FORMAL generates proofs
    5. TC-VERIFIER checks against evidence
    6. TC-PCN binds verified values
    """
    
    # Simulated extraction result
    extraction = {
        "field": "relative_risk",
        "value": 0.556,
        "trajectory": {
            "steps": [
                {"type": "EXTRACTION", "output": 156, "source": "n_treatment"},
                {"type": "EXTRACTION", "output": 151, "source": "n_control"},
                {"type": "EXTRACTION", "output": 23, "source": "events_treatment"},
                {"type": "EXTRACTION", "output": 40, "source": "events_control"},
                {"type": "CALCULATION", "formula": "(23/156)/(40/151)", "output": 0.556}
            ]
        }
    }
    
    # Simulated verification results from each extension
    verification_layers = {
        "tc_rivals": {
            "consensus_type": "UNANIMOUS",
            "agreement_ratio": 1.0,
            "extractors": 3,
            "vetoes": 0
        },
        "tc_prm": {
            "quality": "GOOD",
            "trajectory_score": 0.85,
            "trustworthy": True
        },
        "tc_qwed": {
            "status": "VERIFIED",
            "engine": "MATH",
            "verified_value": 0.556
        },
        "tc_formal": {
            "status": "PROVED",
            "machine_checkable": True,
            "certificate_hash": "abc123..."
        },
        "tc_verifier": {
            "verdict": "SUPPORTED",
            "confidence": 0.9
        },
        "tc_pcn": {
            "claim_bound": True,
            "verified": True,
            "provenance_uri": "https://clinicaltrials.gov/NCT..."
        }
    }
    
    # Determine final status
    all_passed = all(
        layer.get("status") in ["VERIFIED", "PROVED", None] and
        layer.get("verdict") in ["SUPPORTED", None] and
        layer.get("trustworthy", True)
        for layer in verification_layers.values()
    )
    
    result = IntegratedVerificationResult(
        task_id=hashlib.sha256(b"example").hexdigest()[:12],
        field_name=extraction["field"],
        extracted_value=extraction["value"],
        verification_layers=verification_layers,
        final_status="verified" if all_passed else "flagged",
        confidence=0.92,
        issues=[],
        provenance={
            "pipeline_version": "1.0",
            "extensions_used": list(verification_layers.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return result


def print_verification_report(result: IntegratedVerificationResult):
    """Print a formatted verification report."""
    
    status_icons = {
        "verified": "✓",
        "flagged": "⚠",
        "rejected": "✗"
    }
    
    print("\n" + "="*60)
    print("TRUTHCERT INTEGRATED VERIFICATION REPORT")
    print("="*60)
    
    icon = status_icons.get(result.final_status, "?")
    print(f"\n{icon} {result.field_name}: {result.extracted_value}")
    print(f"   Status: {result.final_status.upper()}")
    print(f"   Confidence: {result.confidence:.1%}")
    
    print("\n" + "-"*60)
    print("VERIFICATION LAYERS")
    print("-"*60)
    
    layer_icons = {
        "VERIFIED": "✓", "PROVED": "✓", "SUPPORTED": "✓",
        "GOOD": "✓", "OPTIMAL": "✓", "UNANIMOUS": "✓",
        "FLAGGED": "⚠", "UNKNOWN": "?",
        "REJECTED": "✗", "REFUTED": "✗", "DISPROVED": "✗"
    }
    
    for layer, data in result.verification_layers.items():
        status = (data.get("status") or data.get("verdict") or 
                  data.get("quality") or data.get("consensus_type") or "OK")
        icon = layer_icons.get(status, "○")
        print(f"  {icon} {layer}: {status}")
    
    if result.issues:
        print("\n" + "-"*60)
        print("ISSUES")
        print("-"*60)
        for issue in result.issues:
            print(f"  ⚠ {issue}")
    
    print("\n" + "-"*60)
    print(f"Task ID: {result.task_id}")
    print("="*60)


# =============================================================================
# RECOMMENDED EXTENSION COMBINATIONS
# =============================================================================

EXTENSION_COMBOS = {
    "rct_extraction": {
        "description": "Extract data from RCT reports",
        "primary": ["TC-PCN", "TC-RIVALS"],
        "secondary": ["TC-QWED", "TC-PRM"],
        "rationale": "Multi-agent extraction with numeric verification"
    },
    "meta_analysis_synthesis": {
        "description": "Combine extracted data for meta-analysis",
        "primary": ["TC-QWED", "TC-FORMAL"],
        "secondary": ["TC-VERIFIER", "TC-PCN"],
        "rationale": "Deterministic verification with formal proofs"
    },
    "complex_reasoning": {
        "description": "Verify multi-step reasoning chains",
        "primary": ["TC-PRM"],
        "secondary": ["TC-RIVALS", "TC-FORMAL"],
        "rationale": "Step-level quality with formal verification"
    },
    "evidence_synthesis": {
        "description": "Synthesize evidence from multiple sources",
        "primary": ["TC-VERIFIER", "TC-RIVALS"],
        "secondary": ["TC-PCN"],
        "rationale": "Evidence cross-checking with multi-agent consensus"
    }
}


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# TruthCert Extension Integration Example")
    print("#"*60)
    
    # Show pipeline config
    config = create_integrated_pipeline()
    print("\nPipeline Configuration:")
    print(json.dumps(config, indent=2))
    
    # Run example
    result = example_verification_workflow()
    print_verification_report(result)
    
    # Show recommended combos
    print("\n" + "="*60)
    print("RECOMMENDED EXTENSION COMBINATIONS")
    print("="*60)
    
    for use_case, combo in EXTENSION_COMBOS.items():
        print(f"\n{use_case}:")
        print(f"  {combo['description']}")
        print(f"  Primary: {', '.join(combo['primary'])}")
        print(f"  Secondary: {', '.join(combo['secondary'])}")

"""
TC-VERIFIER: CompassVerifier-Based Verification Extension
==========================================================

Based on: "CompassVerifier: Operationalizing a Practical Verifier for LLM Generation"
(arXiv:2508.03686, EMNLP 2025)

Key Features:
- Unified verifier for math, reasoning, and knowledge claims
- Three-label classification: SUPPORTED / REFUTED / NOT ENOUGH INFO
- Handles multiple answer types: binary, extractive, categorical, numeric
- Trained on VerifierBench across 5 domains

TruthCert Integration:
- Acts as secondary verification layer after extraction
- Provides confidence-weighted verdicts
- Can verify against retrieved sources
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime
import hashlib
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TC-VERIFIER")


# =============================================================================
# ENUMS
# =============================================================================

class VerifierVerdict(Enum):
    """Three-label classification from CompassVerifier."""
    SUPPORTED = auto()         # Claim is verified by evidence
    REFUTED = auto()           # Claim contradicts evidence
    NOT_ENOUGH_INFO = auto()   # Cannot determine from available evidence


class ClaimType(Enum):
    """Types of claims that can be verified."""
    BINARY = auto()            # Yes/No, True/False
    EXTRACTIVE = auto()        # Specific span from text
    CATEGORICAL = auto()       # One of several options
    NUMERIC = auto()           # Numerical value
    REASONING = auto()         # Multi-step reasoning chain
    MATH = auto()              # Mathematical calculation


class VerificationDomain(Enum):
    """Domains supported by CompassVerifier."""
    MATH = "math"
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    CODE = "code"
    SCIENCE = "science"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Evidence:
    """A piece of evidence for verification."""
    evidence_id: str
    content: str                       # The evidence text
    source: str                        # Where this evidence came from
    source_type: str                   # "document", "database", "web", "calculation"
    relevance_score: float = 1.0       # How relevant to the claim (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = hashlib.sha256(
                f"{self.content[:100]}:{self.source}".encode()
            ).hexdigest()[:12]


@dataclass
class Claim:
    """A claim to be verified."""
    claim_id: str
    claim_text: str                    # Natural language claim
    claim_type: ClaimType
    domain: VerificationDomain
    expected_value: Optional[Any] = None  # For numeric/categorical claims
    context: Optional[str] = None      # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.sha256(
                f"{self.claim_text}:{self.claim_type.name}".encode()
            ).hexdigest()[:12]


@dataclass
class VerificationResult:
    """Result of verifying a claim."""
    result_id: str
    claim_id: str
    verdict: VerifierVerdict
    confidence: float                  # 0-1 confidence in verdict
    explanation: str                   # Why this verdict was reached
    supporting_evidence: List[str]     # evidence_ids that support
    contradicting_evidence: List[str]  # evidence_ids that contradict
    reasoning_trace: Optional[str] = None  # Step-by-step verification
    verified_value: Optional[Any] = None   # For numeric claims, the verified value
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.result_id:
            self.result_id = hashlib.sha256(
                f"{self.claim_id}:{self.verdict.name}:{self.timestamp.isoformat()}".encode()
            ).hexdigest()[:12]


@dataclass
class VerificationBatch:
    """Batch of claims verified together."""
    batch_id: str
    claims: List[Claim]
    results: List[VerificationResult]
    evidence_pool: List[Evidence]
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# VERIFIER ENGINES
# =============================================================================

class BaseVerifierEngine:
    """Base class for domain-specific verifiers."""
    
    def __init__(self, domain: VerificationDomain):
        self.domain = domain
    
    def verify(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        raise NotImplementedError


class MathVerifierEngine(BaseVerifierEngine):
    """Verifier for mathematical claims."""
    
    def __init__(self):
        super().__init__(VerificationDomain.MATH)
    
    def verify(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        """
        Verify mathematical claims.
        
        Strategies:
        1. Parse and evaluate expressions
        2. Compare with expected value
        3. Check calculation steps
        """
        try:
            # Try to evaluate the claim mathematically
            # Extract numbers and operations from claim
            
            # Example: "2 + 2 = 4" -> evaluate left side, compare to right
            equals_match = re.search(r'(.+?)\s*=\s*(.+?)(?:\s|$|,|\.)', claim.claim_text)
            
            if equals_match:
                left_expr = equals_match.group(1).strip()
                right_value = equals_match.group(2).strip()
                
                # Safely evaluate (limited to basic math)
                # In production, use sympy or a proper expression parser
                try:
                    # Remove any non-math characters for safety
                    safe_left = re.sub(r'[^0-9+\-*/().^ ]', '', left_expr)
                    safe_left = safe_left.replace('^', '**')
                    
                    computed = eval(safe_left)  # In production, use safer evaluation
                    expected = float(right_value)
                    
                    if abs(computed - expected) < 1e-9:
                        return VerificationResult(
                            result_id="",
                            claim_id=claim.claim_id,
                            verdict=VerifierVerdict.SUPPORTED,
                            confidence=1.0,
                            explanation=f"Mathematical verification: {left_expr} = {computed} ≈ {expected}",
                            supporting_evidence=[],
                            contradicting_evidence=[],
                            reasoning_trace=f"Evaluated '{safe_left}' = {computed}",
                            verified_value=computed
                        )
                    else:
                        return VerificationResult(
                            result_id="",
                            claim_id=claim.claim_id,
                            verdict=VerifierVerdict.REFUTED,
                            confidence=1.0,
                            explanation=f"Mathematical error: {left_expr} = {computed}, not {expected}",
                            supporting_evidence=[],
                            contradicting_evidence=[],
                            reasoning_trace=f"Evaluated '{safe_left}' = {computed}, expected {expected}",
                            verified_value=computed
                        )
                except Exception as e:
                    pass  # Fall through to NOT_ENOUGH_INFO
            
            # If we have expected value, try to verify against evidence
            if claim.expected_value is not None:
                for ev in evidence:
                    if str(claim.expected_value) in ev.content:
                        return VerificationResult(
                            result_id="",
                            claim_id=claim.claim_id,
                            verdict=VerifierVerdict.SUPPORTED,
                            confidence=0.8,
                            explanation=f"Value {claim.expected_value} found in evidence",
                            supporting_evidence=[ev.evidence_id],
                            contradicting_evidence=[],
                            verified_value=claim.expected_value
                        )
            
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                confidence=0.5,
                explanation="Could not mathematically verify claim",
                supporting_evidence=[],
                contradicting_evidence=[]
            )
            
        except Exception as e:
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                explanation=f"Verification error: {str(e)}",
                supporting_evidence=[],
                contradicting_evidence=[]
            )


class KnowledgeVerifierEngine(BaseVerifierEngine):
    """Verifier for factual knowledge claims."""
    
    def __init__(self):
        super().__init__(VerificationDomain.KNOWLEDGE)
    
    def verify(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        """
        Verify factual claims against evidence.
        
        Uses:
        1. Exact match in evidence
        2. Semantic similarity (placeholder for embedding-based)
        3. Contradiction detection
        """
        if not evidence:
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                explanation="No evidence provided for verification",
                supporting_evidence=[],
                contradicting_evidence=[]
            )
        
        supporting = []
        contradicting = []
        
        # Normalize claim for comparison
        claim_lower = claim.claim_text.lower()
        claim_tokens = set(re.findall(r'\b\w+\b', claim_lower))
        
        for ev in evidence:
            ev_lower = ev.content.lower()
            ev_tokens = set(re.findall(r'\b\w+\b', ev_lower))
            
            # Calculate simple overlap
            overlap = len(claim_tokens & ev_tokens) / len(claim_tokens) if claim_tokens else 0
            
            if overlap > 0.5:
                # Check for negation patterns
                negation_patterns = [
                    r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bfalse\b',
                    r'\bincorrect\b', r'\bwrong\b', r'\buntrue\b'
                ]
                
                claim_has_negation = any(re.search(p, claim_lower) for p in negation_patterns)
                ev_has_negation = any(re.search(p, ev_lower) for p in negation_patterns)
                
                if claim_has_negation != ev_has_negation:
                    contradicting.append(ev.evidence_id)
                else:
                    supporting.append(ev.evidence_id)
        
        # Determine verdict
        if supporting and not contradicting:
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.SUPPORTED,
                confidence=min(0.9, 0.5 + 0.1 * len(supporting)),
                explanation=f"Claim supported by {len(supporting)} evidence source(s)",
                supporting_evidence=supporting,
                contradicting_evidence=[]
            )
        elif contradicting and not supporting:
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.REFUTED,
                confidence=min(0.9, 0.5 + 0.1 * len(contradicting)),
                explanation=f"Claim contradicted by {len(contradicting)} evidence source(s)",
                supporting_evidence=[],
                contradicting_evidence=contradicting
            )
        elif supporting and contradicting:
            # Conflicting evidence
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                confidence=0.3,
                explanation=f"Conflicting evidence: {len(supporting)} supporting, {len(contradicting)} contradicting",
                supporting_evidence=supporting,
                contradicting_evidence=contradicting
            )
        else:
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                confidence=0.2,
                explanation="No relevant evidence found",
                supporting_evidence=[],
                contradicting_evidence=[]
            )


class ReasoningVerifierEngine(BaseVerifierEngine):
    """Verifier for multi-step reasoning chains."""
    
    def __init__(self):
        super().__init__(VerificationDomain.REASONING)
    
    def verify(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        """
        Verify reasoning claims by checking logical steps.
        
        Placeholder for actual reasoning verification.
        In production, would use:
        - Chain-of-thought decomposition
        - Step-by-step validation
        - Logical consistency checking
        """
        # Split claim into steps if possible
        steps = re.split(r'(?:therefore|thus|hence|so|because|since|then)\s+', 
                        claim.claim_text, flags=re.IGNORECASE)
        
        if len(steps) > 1:
            # Multi-step reasoning detected
            return VerificationResult(
                result_id="",
                claim_id=claim.claim_id,
                verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                confidence=0.5,
                explanation=f"Detected {len(steps)}-step reasoning chain. Full verification requires LLM analysis.",
                supporting_evidence=[],
                contradicting_evidence=[],
                reasoning_trace=f"Steps identified: {steps}"
            )
        else:
            # Single claim, delegate to knowledge verifier
            knowledge_engine = KnowledgeVerifierEngine()
            return knowledge_engine.verify(claim, evidence)


class NumericVerifierEngine(BaseVerifierEngine):
    """Verifier specifically for numeric claims."""
    
    def __init__(self, tolerance: float = 0.01):
        super().__init__(VerificationDomain.SCIENCE)
        self.tolerance = tolerance
    
    def verify(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        """
        Verify numeric claims with tolerance.
        
        Handles:
        - Exact values
        - Ranges
        - Percentages
        - Scientific notation
        """
        if claim.expected_value is None:
            # Try to extract expected value from claim
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', claim.claim_text)
            if not numbers:
                return VerificationResult(
                    result_id="",
                    claim_id=claim.claim_id,
                    verdict=VerifierVerdict.NOT_ENOUGH_INFO,
                    confidence=0.0,
                    explanation="No numeric value found in claim",
                    supporting_evidence=[],
                    contradicting_evidence=[]
                )
            expected = float(numbers[-1])  # Use last number as the claim value
        else:
            expected = float(claim.expected_value)
        
        # Search evidence for matching numbers
        for ev in evidence:
            ev_numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', ev.content)
            
            for num_str in ev_numbers:
                try:
                    found = float(num_str)
                    
                    # Check with tolerance
                    if expected != 0:
                        relative_diff = abs(found - expected) / abs(expected)
                        if relative_diff <= self.tolerance:
                            return VerificationResult(
                                result_id="",
                                claim_id=claim.claim_id,
                                verdict=VerifierVerdict.SUPPORTED,
                                confidence=1.0 - relative_diff,
                                explanation=f"Found matching value {found} (expected {expected}, diff={relative_diff:.2%})",
                                supporting_evidence=[ev.evidence_id],
                                contradicting_evidence=[],
                                verified_value=found
                            )
                    elif abs(found - expected) < 1e-9:
                        return VerificationResult(
                            result_id="",
                            claim_id=claim.claim_id,
                            verdict=VerifierVerdict.SUPPORTED,
                            confidence=1.0,
                            explanation=f"Exact match: {found}",
                            supporting_evidence=[ev.evidence_id],
                            contradicting_evidence=[],
                            verified_value=found
                        )
                except ValueError:
                    continue
        
        return VerificationResult(
            result_id="",
            claim_id=claim.claim_id,
            verdict=VerifierVerdict.NOT_ENOUGH_INFO,
            confidence=0.3,
            explanation=f"Could not find value matching {expected} in evidence",
            supporting_evidence=[],
            contradicting_evidence=[]
        )


# =============================================================================
# MAIN VERIFIER
# =============================================================================

class TCVerifier:
    """
    Main TC-VERIFIER class.
    
    Unified verifier that routes claims to appropriate domain engines
    and aggregates results.
    """
    
    def __init__(self, custom_engines: Optional[Dict[VerificationDomain, BaseVerifierEngine]] = None):
        # Initialize default engines
        self.engines: Dict[VerificationDomain, BaseVerifierEngine] = {
            VerificationDomain.MATH: MathVerifierEngine(),
            VerificationDomain.KNOWLEDGE: KnowledgeVerifierEngine(),
            VerificationDomain.REASONING: ReasoningVerifierEngine(),
            VerificationDomain.SCIENCE: NumericVerifierEngine(),
        }
        
        # Override with custom engines
        if custom_engines:
            self.engines.update(custom_engines)
        
        # Domain routing based on claim type
        self.type_to_domain = {
            ClaimType.BINARY: VerificationDomain.KNOWLEDGE,
            ClaimType.EXTRACTIVE: VerificationDomain.KNOWLEDGE,
            ClaimType.CATEGORICAL: VerificationDomain.KNOWLEDGE,
            ClaimType.NUMERIC: VerificationDomain.SCIENCE,
            ClaimType.REASONING: VerificationDomain.REASONING,
            ClaimType.MATH: VerificationDomain.MATH,
        }
        
        self.verification_count = 0
    
    def _select_engine(self, claim: Claim) -> BaseVerifierEngine:
        """Select appropriate engine for claim."""
        # First try explicit domain
        if claim.domain in self.engines:
            return self.engines[claim.domain]
        
        # Then route by claim type
        domain = self.type_to_domain.get(claim.claim_type, VerificationDomain.KNOWLEDGE)
        return self.engines.get(domain, self.engines[VerificationDomain.KNOWLEDGE])
    
    def verify_claim(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        """Verify a single claim against evidence."""
        self.verification_count += 1
        
        engine = self._select_engine(claim)
        logger.info(f"Verifying claim '{claim.claim_id}' with {engine.domain.value} engine")
        
        result = engine.verify(claim, evidence)
        
        logger.info(f"Verdict: {result.verdict.name} (confidence: {result.confidence:.2f})")
        return result
    
    def verify_batch(self, claims: List[Claim], evidence: List[Evidence]) -> VerificationBatch:
        """Verify multiple claims against shared evidence pool."""
        results = []
        
        for claim in claims:
            result = self.verify_claim(claim, evidence)
            results.append(result)
        
        # Generate summary
        supported = sum(1 for r in results if r.verdict == VerifierVerdict.SUPPORTED)
        refuted = sum(1 for r in results if r.verdict == VerifierVerdict.REFUTED)
        unknown = sum(1 for r in results if r.verdict == VerifierVerdict.NOT_ENOUGH_INFO)
        
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        
        return VerificationBatch(
            batch_id=hashlib.sha256(
                f"{len(claims)}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:12],
            claims=claims,
            results=results,
            evidence_pool=evidence,
            summary={
                "total_claims": len(claims),
                "supported": supported,
                "refuted": refuted,
                "not_enough_info": unknown,
                "average_confidence": avg_confidence,
                "verification_rate": supported / len(claims) if claims else 0
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        return {
            "total_verifications": self.verification_count,
            "engines_available": list(self.engines.keys())
        }


# =============================================================================
# META-ANALYSIS EXAMPLE
# =============================================================================

def example_meta_analysis_verification():
    """
    Example: Verify extracted meta-analysis data.
    """
    print("\n" + "="*70)
    print("TC-VERIFIER EXAMPLE: Meta-Analysis Data Verification")
    print("="*70)
    
    verifier = TCVerifier()
    
    # Evidence: Original trial report
    evidence = [
        Evidence(
            evidence_id="trial_001",
            content="""
            RESULTS: A total of 156 patients were randomized to colchicine and 
            151 patients to placebo. The primary endpoint (MACE at 30 days) 
            occurred in 23 patients (14.7%) in the colchicine group and 
            40 patients (26.5%) in the placebo group (RR 0.56, 95% CI 0.35-0.89, p=0.014).
            """,
            source="ClinicalTrials.gov NCT12345678",
            source_type="database"
        ),
        Evidence(
            evidence_id="calc_001",
            content="Risk Ratio calculation: (23/156) / (40/151) = 0.147 / 0.265 = 0.556",
            source="Statistical computation",
            source_type="calculation"
        )
    ]
    
    # Claims to verify
    claims = [
        Claim(
            claim_id="claim_n_treatment",
            claim_text="The treatment group had 156 patients",
            claim_type=ClaimType.NUMERIC,
            domain=VerificationDomain.SCIENCE,
            expected_value=156
        ),
        Claim(
            claim_id="claim_n_control",
            claim_text="The control group had 151 patients",
            claim_type=ClaimType.NUMERIC,
            domain=VerificationDomain.SCIENCE,
            expected_value=151
        ),
        Claim(
            claim_id="claim_events_treatment",
            claim_text="23 MACE events occurred in the treatment group",
            claim_type=ClaimType.NUMERIC,
            domain=VerificationDomain.SCIENCE,
            expected_value=23
        ),
        Claim(
            claim_id="claim_rr",
            claim_text="The relative risk was 0.56",
            claim_type=ClaimType.NUMERIC,
            domain=VerificationDomain.SCIENCE,
            expected_value=0.56
        ),
        Claim(
            claim_id="claim_math",
            claim_text="(23/156) / (40/151) = 0.556",
            claim_type=ClaimType.MATH,
            domain=VerificationDomain.MATH
        ),
        Claim(
            claim_id="claim_false_n",
            claim_text="The treatment group had 200 patients",
            claim_type=ClaimType.NUMERIC,
            domain=VerificationDomain.SCIENCE,
            expected_value=200
        ),
    ]
    
    # Verify batch
    batch = verifier.verify_batch(claims, evidence)
    
    # Display results
    print("\n" + "-"*70)
    print("VERIFICATION RESULTS")
    print("-"*70)
    
    for claim, result in zip(batch.claims, batch.results):
        status_icon = {
            VerifierVerdict.SUPPORTED: "✓",
            VerifierVerdict.REFUTED: "✗",
            VerifierVerdict.NOT_ENOUGH_INFO: "?"
        }[result.verdict]
        
        print(f"\n{status_icon} {claim.claim_text[:60]}...")
        print(f"  Verdict: {result.verdict.name}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Explanation: {result.explanation}")
        if result.verified_value is not None:
            print(f"  Verified Value: {result.verified_value}")
    
    print("\n" + "-"*70)
    print("BATCH SUMMARY")
    print("-"*70)
    for key, value in batch.summary.items():
        print(f"  {key}: {value}")
    
    return batch


def example_reasoning_verification():
    """
    Example: Verify a reasoning chain.
    """
    print("\n" + "="*70)
    print("TC-VERIFIER EXAMPLE: Reasoning Chain Verification")
    print("="*70)
    
    verifier = TCVerifier()
    
    evidence = [
        Evidence(
            evidence_id="ev_001",
            content="Colchicine reduces inflammation markers",
            source="Literature review",
            source_type="document"
        ),
        Evidence(
            evidence_id="ev_002",
            content="Inflammation is linked to adverse cardiovascular events",
            source="Literature review",
            source_type="document"
        )
    ]
    
    claim = Claim(
        claim_id="reasoning_001",
        claim_text="Colchicine reduces inflammation, therefore it may reduce cardiovascular events",
        claim_type=ClaimType.REASONING,
        domain=VerificationDomain.REASONING
    )
    
    result = verifier.verify_claim(claim, evidence)
    
    print(f"\nClaim: {claim.claim_text}")
    print(f"Verdict: {result.verdict.name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Explanation: {result.explanation}")
    if result.reasoning_trace:
        print(f"Reasoning Trace: {result.reasoning_trace}")
    
    return result


# =============================================================================
# INTEGRATION WITH TRUTHCERT
# =============================================================================

def integrate_with_truthcert(result: VerificationResult, claim: Claim) -> Dict[str, Any]:
    """
    Convert TC-VERIFIER result to TruthCert-compatible format.
    """
    # Map verdict to TruthCert status
    status_map = {
        VerifierVerdict.SUPPORTED: "verified",
        VerifierVerdict.REFUTED: "rejected",
        VerifierVerdict.NOT_ENOUGH_INFO: "flagged"
    }
    
    return {
        "truthcert_version": "1.0",
        "extension": "TC-VERIFIER",
        "claim": {
            "id": claim.claim_id,
            "text": claim.claim_text,
            "type": claim.claim_type.name,
            "domain": claim.domain.value
        },
        "verification": {
            "status": status_map[result.verdict],
            "verdict": result.verdict.name,
            "confidence": result.confidence,
            "explanation": result.explanation
        },
        "evidence": {
            "supporting": result.supporting_evidence,
            "contradicting": result.contradicting_evidence
        },
        "provenance": {
            "result_id": result.result_id,
            "timestamp": result.timestamp.isoformat(),
            "verified_value": result.verified_value
        },
        "audit_hash": hashlib.sha256(
            json.dumps({
                "claim_id": claim.claim_id,
                "verdict": result.verdict.name,
                "confidence": result.confidence
            }, sort_keys=True).encode()
        ).hexdigest()[:16]
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# TC-VERIFIER: CompassVerifier-Based Verification Extension")
    print("# TruthCert Protocol - Unified Claim Verification")
    print("#"*70)
    
    # Example 1: Meta-analysis verification
    batch = example_meta_analysis_verification()
    
    # Example 2: Reasoning verification
    reasoning_result = example_reasoning_verification()
    
    # Show TruthCert integration
    print("\n" + "="*70)
    print("TRUTHCERT INTEGRATION FORMAT")
    print("="*70)
    
    tc_format = integrate_with_truthcert(batch.results[0], batch.claims[0])
    print(json.dumps(tc_format, indent=2))
    
    print("\n" + "="*70)
    print("TC-VERIFIER Implementation Complete")
    print("="*70)

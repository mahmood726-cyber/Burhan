"""
TC-PCN: Proof-Carrying Numbers Extension for TruthCert
=======================================================

Implements the Proof-Carrying Numbers protocol for numeric verification
in TruthCert bundles. Based on arXiv:2509.06902.

This extension enforces numeric fidelity through mechanical verification:
- Numeric spans are emitted as claim-bound tokens
- A verifier checks each token under a declared policy
- Only verified numbers receive provenance marks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import hashlib
import json
import re


class VerificationPolicy(Enum):
    """Policies for verifying numeric claims."""
    EXACT = "exact"           # Bit-identical match
    ROUNDED = "rounded"       # Match within decimal places
    TOLERANCE = "tolerance"   # Match within epsilon
    ALIAS = "alias"          # Match known equivalents
    RANGE = "range"          # Value within bounds


class VerificationStatus(Enum):
    """Status of a verified numeric span."""
    VERIFIED = "verified"     # Mechanically checked, passed
    BARE = "bare"            # No claim binding provided
    FLAGGED = "flagged"      # Claim binding failed verification


@dataclass
class ClaimSource:
    """Source metadata for a claim."""
    source_type: str         # e.g., "clinicaltrials.gov", "pubmed"
    source_id: str          # e.g., "NCT03661411"
    field_path: str         # e.g., "primary_outcome.events"
    accessed_at: datetime
    raw_value: Any          # Original value from source
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.source_type,
            "id": self.source_id,
            "field": self.field_path,
            "accessed": self.accessed_at.isoformat(),
            "raw": self.raw_value
        }


@dataclass
class Claim:
    """A structured claim with value and provenance."""
    claim_id: str
    value: Any
    value_type: str         # "integer", "float", "ratio", "interval"
    source: ClaimSource
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def hash(self) -> str:
        """Generate content hash for provenance."""
        content = json.dumps({
            "id": self.claim_id,
            "value": str(self.value),
            "source": self.source.to_dict()
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ClaimBoundToken:
    """A numeric span bound to a structured claim."""
    span: str              # The numeric string as it appears
    claim_id: str          # Reference to the claim
    policy: VerificationPolicy
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    
    # For ROUNDED: decimal_places
    # For TOLERANCE: epsilon
    # For RANGE: lower_bound, upper_bound


@dataclass
class VerificationResult:
    """Result of verifying a claim-bound token."""
    status: VerificationStatus
    span: str
    claim_id: Optional[str] = None
    provenance_hash: Optional[str] = None
    badge_metadata: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None
    expected_value: Optional[Any] = None
    received_value: Optional[Any] = None


class PolicyChecker:
    """Implements verification logic for each policy type."""
    
    @staticmethod
    def check_exact(span: str, claim_value: Any) -> bool:
        """Exact equality check."""
        try:
            if isinstance(claim_value, int):
                return int(span) == claim_value
            elif isinstance(claim_value, float):
                return float(span) == claim_value
            else:
                return str(span).strip() == str(claim_value).strip()
        except ValueError:
            return False
    
    @staticmethod
    def check_rounded(span: str, claim_value: float, decimal_places: int) -> bool:
        """Check equality after rounding."""
        try:
            span_rounded = round(float(span), decimal_places)
            claim_rounded = round(float(claim_value), decimal_places)
            return span_rounded == claim_rounded
        except ValueError:
            return False
    
    @staticmethod
    def check_tolerance(span: str, claim_value: float, epsilon: float) -> bool:
        """Check within tolerance."""
        try:
            return abs(float(span) - float(claim_value)) <= epsilon
        except ValueError:
            return False
    
    @staticmethod
    def check_alias(span: str, claim_value: Any, aliases: List[str]) -> bool:
        """Check against known equivalents."""
        normalized_span = span.strip().lower()
        all_forms = [str(claim_value).lower()] + [a.lower() for a in aliases]
        return normalized_span in all_forms
    
    @staticmethod
    def check_range(span: str, lower: float, upper: float) -> bool:
        """Check value within bounds."""
        try:
            val = float(span)
            return lower <= val <= upper
        except ValueError:
            return False


class TCPCNVerifier:
    """
    Main verifier for Proof-Carrying Numbers in TruthCert.
    
    Implements the rendering contract: only VERIFIED numbers
    receive provenance marks; all others are BARE or FLAGGED.
    """
    
    def __init__(self, claim_store: Optional[Dict[str, Claim]] = None):
        self.claim_store = claim_store or {}
        self.checker = PolicyChecker()
    
    def register_claim(self, claim: Claim) -> None:
        """Register a claim for verification."""
        self.claim_store[claim.claim_id] = claim
    
    def verify_token(self, token: ClaimBoundToken) -> VerificationResult:
        """
        Verify a claim-bound token against its referenced claim.
        
        Returns VerificationResult with:
        - VERIFIED if check passes
        - FLAGGED if check fails
        - BARE if no claim is referenced
        """
        # Handle bare tokens
        if not token.claim_id:
            return VerificationResult(
                status=VerificationStatus.BARE,
                span=token.span
            )
        
        # Look up claim
        claim = self.claim_store.get(token.claim_id)
        if not claim:
            return VerificationResult(
                status=VerificationStatus.FLAGGED,
                span=token.span,
                claim_id=token.claim_id,
                failure_reason=f"Claim '{token.claim_id}' not found in store"
            )
        
        # Run appropriate check
        passed = self._check_policy(token, claim)
        
        if passed:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                span=token.span,
                claim_id=token.claim_id,
                provenance_hash=claim.hash(),
                badge_metadata={
                    "source": claim.source.source_type,
                    "policy": token.policy.value,
                    "verified_at": datetime.utcnow().isoformat()
                }
            )
        else:
            return VerificationResult(
                status=VerificationStatus.FLAGGED,
                span=token.span,
                claim_id=token.claim_id,
                failure_reason=f"Value mismatch under {token.policy.value} policy",
                expected_value=claim.value,
                received_value=token.span
            )
    
    def _check_policy(self, token: ClaimBoundToken, claim: Claim) -> bool:
        """Execute policy-specific verification."""
        if token.policy == VerificationPolicy.EXACT:
            return self.checker.check_exact(token.span, claim.value)
        
        elif token.policy == VerificationPolicy.ROUNDED:
            dp = token.qualifiers.get("decimal_places", 2)
            return self.checker.check_rounded(token.span, claim.value, dp)
        
        elif token.policy == VerificationPolicy.TOLERANCE:
            eps = token.qualifiers.get("epsilon", 0.01)
            return self.checker.check_tolerance(token.span, claim.value, eps)
        
        elif token.policy == VerificationPolicy.ALIAS:
            aliases = token.qualifiers.get("aliases", [])
            return self.checker.check_alias(token.span, claim.value, aliases)
        
        elif token.policy == VerificationPolicy.RANGE:
            lower = token.qualifiers.get("lower_bound", float("-inf"))
            upper = token.qualifiers.get("upper_bound", float("inf"))
            return self.checker.check_range(token.span, lower, upper)
        
        return False
    
    def verify_extraction(
        self,
        extraction: Dict[str, Any],
        claim_bindings: Dict[str, ClaimBoundToken]
    ) -> Dict[str, VerificationResult]:
        """
        Verify all numeric fields in an extraction.
        
        Args:
            extraction: Dictionary of field_name -> value
            claim_bindings: Dictionary of field_name -> ClaimBoundToken
            
        Returns:
            Dictionary of field_name -> VerificationResult
        """
        results = {}
        
        for field_name, value in extraction.items():
            # Skip non-numeric fields
            if not self._is_numeric(value):
                continue
            
            # Get binding or create bare token
            token = claim_bindings.get(field_name)
            if token is None:
                token = ClaimBoundToken(
                    span=str(value),
                    claim_id=None,
                    policy=VerificationPolicy.EXACT
                )
            
            results[field_name] = self.verify_token(token)
        
        return results
    
    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Check if a value is numeric or numeric string."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                # Check for ratio format like "23/156"
                if re.match(r'^\d+/\d+$', value.strip()):
                    return True
        return False


class TCPCNRenderer:
    """
    Rendering component that displays verification status.
    
    Implements the fail-closed contract: unverified numbers
    are visually distinct from verified ones.
    """
    
    BADGE_VERIFIED = "✓"
    BADGE_FLAGGED = "⚠"
    BADGE_BARE = ""
    
    def render_result(
        self,
        result: VerificationResult,
        format: str = "text"
    ) -> str:
        """
        Render a verification result with appropriate marking.
        
        Args:
            result: VerificationResult to render
            format: Output format ("text", "html", "markdown")
            
        Returns:
            Formatted string with verification badge
        """
        if format == "text":
            return self._render_text(result)
        elif format == "html":
            return self._render_html(result)
        elif format == "markdown":
            return self._render_markdown(result)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _render_text(self, result: VerificationResult) -> str:
        if result.status == VerificationStatus.VERIFIED:
            source = result.badge_metadata.get("source", "unknown")
            return f"{result.span} {self.BADGE_VERIFIED}[{source}]"
        elif result.status == VerificationStatus.FLAGGED:
            return f"{result.span} {self.BADGE_FLAGGED}[{result.failure_reason}]"
        else:
            return result.span
    
    def _render_html(self, result: VerificationResult) -> str:
        if result.status == VerificationStatus.VERIFIED:
            source = result.badge_metadata.get("source", "unknown")
            return (
                f'<span class="tc-pcn-verified" '
                f'data-claim-id="{result.claim_id}" '
                f'data-provenance="{result.provenance_hash}" '
                f'title="Verified from {source}">'
                f'{result.span} ✓</span>'
            )
        elif result.status == VerificationStatus.FLAGGED:
            return (
                f'<span class="tc-pcn-flagged" '
                f'title="{result.failure_reason}">'
                f'{result.span} ⚠</span>'
            )
        else:
            return f'<span class="tc-pcn-bare">{result.span}</span>'
    
    def _render_markdown(self, result: VerificationResult) -> str:
        if result.status == VerificationStatus.VERIFIED:
            return f"**{result.span}** ✓"
        elif result.status == VerificationStatus.FLAGGED:
            return f"~~{result.span}~~ ⚠"
        else:
            return result.span


# Example usage for meta-analysis
def example_meta_analysis_verification():
    """Demonstrate TC-PCN for RCT data extraction."""
    
    # Initialize verifier
    verifier = TCPCNVerifier()
    
    # Register claims from ClinicalTrials.gov
    claim1 = Claim(
        claim_id="NCT03661411-colchicine-events",
        value=23,
        value_type="integer",
        source=ClaimSource(
            source_type="clinicaltrials.gov",
            source_id="NCT03661411",
            field_path="OutcomeMeasure.PrimaryOutcome.Events.Intervention",
            accessed_at=datetime.utcnow(),
            raw_value=23
        )
    )
    
    claim2 = Claim(
        claim_id="NCT03661411-colchicine-total",
        value=156,
        value_type="integer",
        source=ClaimSource(
            source_type="clinicaltrials.gov",
            source_id="NCT03661411",
            field_path="EnrollmentCount.InterventionArm",
            accessed_at=datetime.utcnow(),
            raw_value=156
        )
    )
    
    verifier.register_claim(claim1)
    verifier.register_claim(claim2)
    
    # Create claim-bound tokens from extraction
    tokens = [
        ClaimBoundToken(
            span="23",
            claim_id="NCT03661411-colchicine-events",
            policy=VerificationPolicy.EXACT
        ),
        ClaimBoundToken(
            span="156",
            claim_id="NCT03661411-colchicine-total",
            policy=VerificationPolicy.EXACT
        ),
        ClaimBoundToken(
            span="0.73",
            claim_id=None,  # No claim binding for derived value
            policy=VerificationPolicy.EXACT
        )
    ]
    
    # Verify and render
    renderer = TCPCNRenderer()
    
    print("TC-PCN Verification Results:")
    print("-" * 40)
    
    for token in tokens:
        result = verifier.verify_token(token)
        rendered = renderer.render_result(result, format="text")
        print(f"  {rendered}")
        print(f"    Status: {result.status.value}")
        if result.provenance_hash:
            print(f"    Provenance: {result.provenance_hash}")
        print()


if __name__ == "__main__":
    example_meta_analysis_verification()

"""
TC-RIVALS: Team of Rivals Multi-Agent Verification Extension
=============================================================

Based on: "Team of Rivals: Competitive Multi-Agent System for LLMs with Veto-Based Checking"
(arXiv:2601.14351)

Architecture:
- Controller (brain): Task decomposition, orchestration, no direct extraction
- Extractors (hands): Independent execution, multiple agents, no veto power
- Critic: Verification specialist with VETO authority
- Arbitrator: Final decision authority on conflicts

Key Innovation: >90% error interception through role separation + veto authority

TruthCert Integration:
- Multi-witness requirement maps to multiple extractors
- Critic veto maps to verification gating
- Arbitration maps to conflict resolution protocols
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime
import hashlib
import json
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TC-RIVALS")


# =============================================================================
# ENUMS
# =============================================================================

class AgentRole(Enum):
    """Roles in the Team of Rivals architecture."""
    CONTROLLER = "controller"      # Task decomposition, orchestration
    EXTRACTOR = "extractor"        # Independent data extraction
    CRITIC = "critic"              # Verification with veto power
    ARBITRATOR = "arbitrator"      # Final conflict resolution


class ExtractionStatus(Enum):
    """Status of an extraction attempt."""
    PENDING = auto()
    COMPLETE = auto()
    FAILED = auto()
    VETOED = auto()


class ConsensusType(Enum):
    """Types of consensus outcomes."""
    UNANIMOUS = auto()         # All extractors agree
    MAJORITY = auto()          # >50% agreement
    PLURALITY = auto()         # Largest group agrees, no majority
    SPLIT = auto()             # No clear agreement
    SINGLE = auto()            # Only one extractor succeeded


class CriticVerdict(Enum):
    """Critic's verdict on extractions."""
    APPROVE = auto()           # Extraction passes verification
    FLAG = auto()              # Concern noted but not blocking
    VETO = auto()              # Extraction rejected, triggers re-extraction or arbitration


class ArbitrationOutcome(Enum):
    """Arbitrator's final decision."""
    ACCEPT_CONSENSUS = auto()  # Accept the consensus value
    ACCEPT_MINORITY = auto()   # Accept a minority position (with justification)
    ACCEPT_CORRECTED = auto()  # Accept a corrected/merged value
    REJECT_ALL = auto()        # No acceptable extraction, mark as unverified
    ESCALATE = auto()          # Requires human review


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractionTask:
    """A single extraction task assigned to extractors."""
    task_id: str
    field_name: str                    # e.g., "n_treatment", "mean_control", "outcome"
    source_context: str                # The text/data to extract from
    extraction_type: str               # "numeric", "categorical", "text", "date"
    validation_hints: Dict[str, Any] = field(default_factory=dict)  # Expected ranges, formats
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.task_id:
            # Generate deterministic ID from content
            content = f"{self.field_name}:{self.source_context[:100]}"
            self.task_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class ExtractionValue:
    """A single extracted value from one extractor."""
    value: Any                         # The extracted value
    confidence: float                  # 0.0-1.0 confidence score
    reasoning: str                     # Explanation of extraction logic
    source_span: Optional[str] = None  # The specific text span used
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Extraction:
    """Complete extraction from one agent."""
    extraction_id: str
    task_id: str
    extractor_id: str
    status: ExtractionStatus
    value: Optional[ExtractionValue] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.extraction_id:
            content = f"{self.task_id}:{self.extractor_id}:{self.timestamp.isoformat()}"
            self.extraction_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class CriticFlag:
    """A concern raised by the critic."""
    flag_id: str
    extraction_ids: List[str]          # Which extractions this flag applies to
    flag_type: str                     # "value_mismatch", "implausible", "calculation_error", etc.
    severity: CriticVerdict
    description: str
    suggested_correction: Optional[Any] = None
    evidence: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConsensusResult:
    """Result of consensus analysis across extractors."""
    consensus_type: ConsensusType
    consensus_value: Optional[Any]     # The agreed-upon value (if any)
    agreement_ratio: float             # Proportion of extractors agreeing
    value_groups: Dict[str, List[str]] # value_hash -> list of extractor_ids
    confidence: float                  # Aggregate confidence
    disagreement_details: Optional[str] = None


@dataclass
class ArbitrationDecision:
    """Arbitrator's final decision on a disputed extraction."""
    decision_id: str
    task_id: str
    outcome: ArbitrationOutcome
    final_value: Optional[Any]
    justification: str
    considered_values: List[Tuple[str, Any]]  # (extractor_id, value) pairs
    critic_flags: List[CriticFlag]
    human_review_required: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VerificationReport:
    """Complete verification report for a task."""
    task: ExtractionTask
    extractions: List[Extraction]
    consensus: ConsensusResult
    critic_flags: List[CriticFlag]
    arbitration: Optional[ArbitrationDecision]
    final_value: Optional[Any]
    final_status: str                  # "verified", "flagged", "rejected", "human_review"
    provenance: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AGENT IMPLEMENTATIONS
# =============================================================================

class BaseAgent:
    """Base class for all agents in the Team of Rivals."""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.call_count = 0
    
    def log(self, message: str):
        logger.info(f"[{self.role.value}:{self.agent_id}] {message}")


class Extractor(BaseAgent):
    """
    Extractor agent: Independent data extraction without veto power.
    
    In a real implementation, this would wrap an LLM call.
    Here we provide a pluggable interface.
    """
    
    def __init__(self, agent_id: str, extraction_fn: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.EXTRACTOR)
        self.extraction_fn = extraction_fn or self._default_extraction
    
    def _default_extraction(self, task: ExtractionTask) -> ExtractionValue:
        """Default extraction (placeholder for LLM call)."""
        # This would be replaced with actual LLM extraction
        raise NotImplementedError(
            f"Extractor {self.agent_id} requires an extraction_fn. "
            "Provide a callable that takes ExtractionTask and returns ExtractionValue."
        )
    
    def extract(self, task: ExtractionTask) -> Extraction:
        """Perform extraction on a task."""
        self.call_count += 1
        self.log(f"Extracting field '{task.field_name}' (attempt #{self.call_count})")
        
        try:
            value = self.extraction_fn(task)
            return Extraction(
                extraction_id="",  # Auto-generated
                task_id=task.task_id,
                extractor_id=self.agent_id,
                status=ExtractionStatus.COMPLETE,
                value=value
            )
        except Exception as e:
            self.log(f"Extraction failed: {str(e)}")
            return Extraction(
                extraction_id="",
                task_id=task.task_id,
                extractor_id=self.agent_id,
                status=ExtractionStatus.FAILED,
                error_message=str(e)
            )


class Critic(BaseAgent):
    """
    Critic agent: Verification specialist with VETO authority.
    
    The critic reviews extractions and can:
    - APPROVE: Extraction is valid
    - FLAG: Note a concern but allow to proceed
    - VETO: Block the extraction, trigger re-extraction or arbitration
    """
    
    def __init__(self, agent_id: str, verification_fn: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.CRITIC)
        self.verification_fn = verification_fn or self._default_verification
        self.veto_count = 0
    
    def _default_verification(
        self, 
        task: ExtractionTask, 
        extractions: List[Extraction],
        consensus: ConsensusResult
    ) -> List[CriticFlag]:
        """Default verification logic."""
        flags = []
        
        # Check 1: Low confidence extractions
        for ext in extractions:
            if ext.value and ext.value.confidence < 0.5:
                flags.append(CriticFlag(
                    flag_id=f"low_conf_{ext.extraction_id}",
                    extraction_ids=[ext.extraction_id],
                    flag_type="low_confidence",
                    severity=CriticVerdict.FLAG,
                    description=f"Extraction confidence ({ext.value.confidence:.2f}) below threshold"
                ))
        
        # Check 2: Split consensus (no agreement)
        if consensus.consensus_type == ConsensusType.SPLIT:
            flags.append(CriticFlag(
                flag_id=f"split_{task.task_id}",
                extraction_ids=[e.extraction_id for e in extractions],
                flag_type="no_consensus",
                severity=CriticVerdict.VETO,
                description=f"No consensus reached. Value groups: {len(consensus.value_groups)}"
            ))
        
        # Check 3: Validation hints
        if consensus.consensus_value is not None and task.validation_hints:
            if "min" in task.validation_hints:
                try:
                    if float(consensus.consensus_value) < task.validation_hints["min"]:
                        flags.append(CriticFlag(
                            flag_id=f"below_min_{task.task_id}",
                            extraction_ids=[e.extraction_id for e in extractions if e.status == ExtractionStatus.COMPLETE],
                            flag_type="implausible_value",
                            severity=CriticVerdict.VETO,
                            description=f"Value {consensus.consensus_value} below minimum {task.validation_hints['min']}"
                        ))
                except (ValueError, TypeError):
                    pass
            
            if "max" in task.validation_hints:
                try:
                    if float(consensus.consensus_value) > task.validation_hints["max"]:
                        flags.append(CriticFlag(
                            flag_id=f"above_max_{task.task_id}",
                            extraction_ids=[e.extraction_id for e in extractions if e.status == ExtractionStatus.COMPLETE],
                            flag_type="implausible_value",
                            severity=CriticVerdict.VETO,
                            description=f"Value {consensus.consensus_value} above maximum {task.validation_hints['max']}"
                        ))
                except (ValueError, TypeError):
                    pass
        
        return flags
    
    def verify(
        self, 
        task: ExtractionTask, 
        extractions: List[Extraction],
        consensus: ConsensusResult
    ) -> Tuple[List[CriticFlag], bool]:
        """
        Verify extractions and return flags.
        
        Returns:
            Tuple of (flags, has_veto)
        """
        self.call_count += 1
        self.log(f"Verifying {len(extractions)} extractions for '{task.field_name}'")
        
        flags = self.verification_fn(task, extractions, consensus)
        
        has_veto = any(f.severity == CriticVerdict.VETO for f in flags)
        if has_veto:
            self.veto_count += 1
            self.log(f"VETO issued! Total vetoes: {self.veto_count}")
        
        return flags, has_veto


class Arbitrator(BaseAgent):
    """
    Arbitrator agent: Final decision authority on conflicts.
    
    Called when:
    - Critic issues a VETO
    - Consensus cannot be reached
    - Values are disputed
    """
    
    def __init__(self, agent_id: str, arbitration_fn: Optional[Callable] = None):
        super().__init__(agent_id, AgentRole.ARBITRATOR)
        self.arbitration_fn = arbitration_fn or self._default_arbitration
    
    def _default_arbitration(
        self,
        task: ExtractionTask,
        extractions: List[Extraction],
        consensus: ConsensusResult,
        flags: List[CriticFlag]
    ) -> ArbitrationDecision:
        """Default arbitration logic."""
        
        # Collect successful extractions
        valid_extractions = [e for e in extractions if e.status == ExtractionStatus.COMPLETE and e.value]
        
        if not valid_extractions:
            return ArbitrationDecision(
                decision_id=f"arb_{task.task_id}",
                task_id=task.task_id,
                outcome=ArbitrationOutcome.REJECT_ALL,
                final_value=None,
                justification="No valid extractions available",
                considered_values=[],
                critic_flags=flags,
                human_review_required=True
            )
        
        # Check if any veto flags have suggested corrections
        veto_flags = [f for f in flags if f.severity == CriticVerdict.VETO]
        for flag in veto_flags:
            if flag.suggested_correction is not None:
                return ArbitrationDecision(
                    decision_id=f"arb_{task.task_id}",
                    task_id=task.task_id,
                    outcome=ArbitrationOutcome.ACCEPT_CORRECTED,
                    final_value=flag.suggested_correction,
                    justification=f"Accepted critic's correction: {flag.description}",
                    considered_values=[(e.extractor_id, e.value.value) for e in valid_extractions],
                    critic_flags=flags
                )
        
        # If we have majority or unanimous, trust it despite veto
        if consensus.consensus_type in [ConsensusType.UNANIMOUS, ConsensusType.MAJORITY]:
            # But flag for human review if there was a veto
            return ArbitrationDecision(
                decision_id=f"arb_{task.task_id}",
                task_id=task.task_id,
                outcome=ArbitrationOutcome.ACCEPT_CONSENSUS,
                final_value=consensus.consensus_value,
                justification=f"Accepting {consensus.consensus_type.name} consensus ({consensus.agreement_ratio:.0%} agreement) despite critic concerns",
                considered_values=[(e.extractor_id, e.value.value) for e in valid_extractions],
                critic_flags=flags,
                human_review_required=len(veto_flags) > 0
            )
        
        # For split/plurality, escalate to human
        return ArbitrationDecision(
            decision_id=f"arb_{task.task_id}",
            task_id=task.task_id,
            outcome=ArbitrationOutcome.ESCALATE,
            final_value=None,
            justification=f"Cannot resolve: {consensus.consensus_type.name} with {len(veto_flags)} veto flags",
            considered_values=[(e.extractor_id, e.value.value) for e in valid_extractions],
            critic_flags=flags,
            human_review_required=True
        )
    
    def arbitrate(
        self,
        task: ExtractionTask,
        extractions: List[Extraction],
        consensus: ConsensusResult,
        flags: List[CriticFlag]
    ) -> ArbitrationDecision:
        """Make final decision on disputed extraction."""
        self.call_count += 1
        self.log(f"Arbitrating task '{task.field_name}' with {len(flags)} flags")
        
        return self.arbitration_fn(task, extractions, consensus, flags)


# =============================================================================
# CONSENSUS ENGINE
# =============================================================================

class ConsensusEngine:
    """Analyzes extraction results to determine consensus."""
    
    @staticmethod
    def value_hash(value: Any) -> str:
        """Create a hash for a value to group identical values."""
        # Handle numeric values with tolerance
        if isinstance(value, (int, float)):
            # Round to 6 decimal places for comparison
            normalized = round(float(value), 6)
            return f"num:{normalized}"
        # Handle strings case-insensitively
        elif isinstance(value, str):
            return f"str:{value.lower().strip()}"
        # Handle None
        elif value is None:
            return "none"
        # Default: JSON serialization
        else:
            try:
                return f"json:{json.dumps(value, sort_keys=True)}"
            except:
                return f"repr:{repr(value)}"
    
    @classmethod
    def analyze(cls, extractions: List[Extraction]) -> ConsensusResult:
        """Analyze extractions to determine consensus."""
        
        # Filter to successful extractions
        valid = [e for e in extractions if e.status == ExtractionStatus.COMPLETE and e.value]
        
        if not valid:
            return ConsensusResult(
                consensus_type=ConsensusType.SPLIT,
                consensus_value=None,
                agreement_ratio=0.0,
                value_groups={},
                confidence=0.0,
                disagreement_details="No valid extractions"
            )
        
        if len(valid) == 1:
            return ConsensusResult(
                consensus_type=ConsensusType.SINGLE,
                consensus_value=valid[0].value.value,
                agreement_ratio=1.0,
                value_groups={cls.value_hash(valid[0].value.value): [valid[0].extractor_id]},
                confidence=valid[0].value.confidence
            )
        
        # Group by value
        value_groups: Dict[str, List[str]] = {}
        value_map: Dict[str, Any] = {}  # hash -> actual value
        confidence_sum: Dict[str, float] = {}
        
        for ext in valid:
            h = cls.value_hash(ext.value.value)
            if h not in value_groups:
                value_groups[h] = []
                value_map[h] = ext.value.value
                confidence_sum[h] = 0.0
            value_groups[h].append(ext.extractor_id)
            confidence_sum[h] += ext.value.confidence
        
        # Find largest group
        largest_hash = max(value_groups.keys(), key=lambda h: len(value_groups[h]))
        largest_count = len(value_groups[largest_hash])
        total_count = len(valid)
        ratio = largest_count / total_count
        
        # Determine consensus type
        if ratio == 1.0:
            consensus_type = ConsensusType.UNANIMOUS
        elif ratio > 0.5:
            consensus_type = ConsensusType.MAJORITY
        elif largest_count > 1:
            consensus_type = ConsensusType.PLURALITY
        else:
            consensus_type = ConsensusType.SPLIT
        
        # Calculate average confidence for consensus group
        avg_confidence = confidence_sum[largest_hash] / largest_count
        
        # Build disagreement details if not unanimous
        disagreement = None
        if consensus_type != ConsensusType.UNANIMOUS:
            details = []
            for h, extractors in value_groups.items():
                details.append(f"{value_map[h]}: {extractors}")
            disagreement = "; ".join(details)
        
        return ConsensusResult(
            consensus_type=consensus_type,
            consensus_value=value_map[largest_hash] if consensus_type != ConsensusType.SPLIT else None,
            agreement_ratio=ratio,
            value_groups=value_groups,
            confidence=avg_confidence,
            disagreement_details=disagreement
        )


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class TCRivalsOrchestrator:
    """
    Main orchestrator for Team of Rivals verification.
    
    Coordinates:
    1. Task distribution to extractors
    2. Consensus analysis
    3. Critic verification
    4. Arbitration (if needed)
    5. Final report generation
    """
    
    def __init__(
        self,
        extractors: List[Extractor],
        critic: Critic,
        arbitrator: Arbitrator,
        min_extractors: int = 2,
        max_retries: int = 2
    ):
        self.extractors = extractors
        self.critic = critic
        self.arbitrator = arbitrator
        self.min_extractors = min_extractors
        self.max_retries = max_retries
        self.consensus_engine = ConsensusEngine()
        
        if len(extractors) < min_extractors:
            raise ValueError(f"Need at least {min_extractors} extractors, got {len(extractors)}")
    
    def _run_extractors(self, task: ExtractionTask) -> List[Extraction]:
        """Run all extractors on a task."""
        extractions = []
        for extractor in self.extractors:
            extraction = extractor.extract(task)
            extractions.append(extraction)
        return extractions
    
    def verify_task(self, task: ExtractionTask) -> VerificationReport:
        """
        Run full verification pipeline on a single task.
        
        Pipeline:
        1. Distribute to extractors
        2. Analyze consensus
        3. Critic review (with possible veto)
        4. Arbitration (if veto or no consensus)
        5. Generate report
        """
        logger.info(f"=== Starting verification for task: {task.field_name} ===")
        
        # Step 1: Run extractors
        extractions = self._run_extractors(task)
        successful = [e for e in extractions if e.status == ExtractionStatus.COMPLETE]
        logger.info(f"Extraction complete: {len(successful)}/{len(extractions)} successful")
        
        # Step 2: Analyze consensus
        consensus = self.consensus_engine.analyze(extractions)
        logger.info(f"Consensus: {consensus.consensus_type.name} ({consensus.agreement_ratio:.0%})")
        
        # Step 3: Critic review
        flags, has_veto = self.critic.verify(task, extractions, consensus)
        logger.info(f"Critic review: {len(flags)} flags, veto={has_veto}")
        
        # Step 4: Arbitration (if needed)
        arbitration = None
        final_value = consensus.consensus_value
        final_status = "verified"
        
        needs_arbitration = (
            has_veto or 
            consensus.consensus_type in [ConsensusType.SPLIT, ConsensusType.PLURALITY]
        )
        
        if needs_arbitration:
            arbitration = self.arbitrator.arbitrate(task, extractions, consensus, flags)
            final_value = arbitration.final_value
            
            if arbitration.outcome == ArbitrationOutcome.REJECT_ALL:
                final_status = "rejected"
            elif arbitration.outcome == ArbitrationOutcome.ESCALATE:
                final_status = "human_review"
            elif arbitration.human_review_required:
                final_status = "flagged"
            else:
                final_status = "verified"
        elif flags:
            final_status = "flagged"
        
        logger.info(f"Final status: {final_status}, value: {final_value}")
        
        # Step 5: Generate report
        return VerificationReport(
            task=task,
            extractions=extractions,
            consensus=consensus,
            critic_flags=flags,
            arbitration=arbitration,
            final_value=final_value,
            final_status=final_status,
            provenance={
                "extractors": [e.agent_id for e in self.extractors],
                "critic": self.critic.agent_id,
                "arbitrator": self.arbitrator.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def verify_batch(self, tasks: List[ExtractionTask]) -> List[VerificationReport]:
        """Verify multiple tasks."""
        reports = []
        for task in tasks:
            report = self.verify_task(task)
            reports.append(report)
        return reports
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "extractor_calls": sum(e.call_count for e in self.extractors),
            "critic_calls": self.critic.call_count,
            "critic_vetoes": self.critic.veto_count,
            "arbitrator_calls": self.arbitrator.call_count,
            "veto_rate": (
                self.critic.veto_count / self.critic.call_count 
                if self.critic.call_count > 0 else 0
            )
        }


# =============================================================================
# META-ANALYSIS EXAMPLE: RCT Data Extraction
# =============================================================================

def create_mock_extractor(extractor_id: str, variance: float = 0.0) -> Extractor:
    """
    Create a mock extractor for testing.
    
    Args:
        extractor_id: Unique identifier
        variance: Amount of random variation to introduce (0.0 = deterministic)
    """
    def extract_fn(task: ExtractionTask) -> ExtractionValue:
        # Simulate extraction from context
        # In real implementation, this would call an LLM
        
        # Parse mock data from context (format: "field=value")
        context = task.source_context
        field = task.field_name
        
        # Look for patterns like "n=156" or "mean=23.5"
        import re
        pattern = rf"{field}\s*[=:]\s*([\d.]+)"
        match = re.search(pattern, context, re.IGNORECASE)
        
        if match:
            value = float(match.group(1))
            # Add variance for testing disagreement
            if variance > 0:
                value += random.uniform(-variance, variance)
                value = round(value, 2)
            
            return ExtractionValue(
                value=value,
                confidence=0.85 + random.uniform(-0.1, 0.1),
                reasoning=f"Extracted from pattern match: {match.group(0)}",
                source_span=match.group(0)
            )
        else:
            raise ValueError(f"Could not find {field} in context")
    
    return Extractor(extractor_id, extract_fn)


def example_rct_extraction():
    """
    Example: Extract RCT data with Team of Rivals verification.
    
    Scenario: Extracting sample sizes from a clinical trial report.
    """
    print("\n" + "="*70)
    print("TC-RIVALS EXAMPLE: RCT Data Extraction")
    print("="*70)
    
    # Create agents
    # Using 3 extractors (odd number helps with consensus)
    extractors = [
        create_mock_extractor("extractor_1", variance=0.0),  # Accurate
        create_mock_extractor("extractor_2", variance=0.0),  # Accurate
        create_mock_extractor("extractor_3", variance=5.0),  # Some variance
    ]
    
    critic = Critic("critic_1")
    arbitrator = Arbitrator("arbitrator_1")
    
    # Create orchestrator
    orchestrator = TCRivalsOrchestrator(
        extractors=extractors,
        critic=critic,
        arbitrator=arbitrator,
        min_extractors=2
    )
    
    # Define extraction tasks
    # Simulated trial report text
    trial_text = """
    METHODS: Patients were randomized 1:1 to colchicine (n_treatment=156) 
    or placebo (n_control=151). The primary endpoint was MACE at 30 days.
    
    RESULTS: In the treatment arm, 23 patients experienced MACE (events_treatment=23).
    In the control arm, 40 patients experienced MACE (events_control=40).
    """
    
    tasks = [
        ExtractionTask(
            task_id="task_n_treatment",
            field_name="n_treatment",
            source_context=trial_text,
            extraction_type="numeric",
            validation_hints={"min": 10, "max": 10000}
        ),
        ExtractionTask(
            task_id="task_n_control",
            field_name="n_control",
            source_context=trial_text,
            extraction_type="numeric",
            validation_hints={"min": 10, "max": 10000}
        ),
        ExtractionTask(
            task_id="task_events_treatment",
            field_name="events_treatment",
            source_context=trial_text,
            extraction_type="numeric",
            validation_hints={"min": 0, "max": 1000}
        ),
        ExtractionTask(
            task_id="task_events_control",
            field_name="events_control",
            source_context=trial_text,
            extraction_type="numeric",
            validation_hints={"min": 0, "max": 1000}
        ),
    ]
    
    # Run verification
    reports = orchestrator.verify_batch(tasks)
    
    # Display results
    print("\n" + "-"*70)
    print("VERIFICATION RESULTS")
    print("-"*70)
    
    for report in reports:
        print(f"\nField: {report.task.field_name}")
        print(f"  Final Value: {report.final_value}")
        print(f"  Status: {report.final_status}")
        print(f"  Consensus: {report.consensus.consensus_type.name} ({report.consensus.agreement_ratio:.0%})")
        
        if report.critic_flags:
            print(f"  Flags:")
            for flag in report.critic_flags:
                print(f"    - [{flag.severity.name}] {flag.description}")
        
        if report.arbitration:
            print(f"  Arbitration: {report.arbitration.outcome.name}")
            print(f"    Justification: {report.arbitration.justification}")
    
    # Show statistics
    print("\n" + "-"*70)
    print("ORCHESTRATOR STATISTICS")
    print("-"*70)
    stats = orchestrator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Calculate error interception rate
    flagged_or_reviewed = sum(1 for r in reports if r.final_status in ["flagged", "human_review", "rejected"])
    print(f"\n  Potential issues flagged: {flagged_or_reviewed}/{len(reports)} tasks")
    
    return reports


def example_with_disagreement():
    """
    Example: Demonstrate handling of extractor disagreement.
    """
    print("\n" + "="*70)
    print("TC-RIVALS EXAMPLE: Handling Disagreement")
    print("="*70)
    
    # Create extractors with intentional disagreement
    def make_biased_extractor(extractor_id: str, bias: float):
        def extract_fn(task: ExtractionTask) -> ExtractionValue:
            # Simulate different interpretations
            base_value = 100
            return ExtractionValue(
                value=base_value + bias,
                confidence=0.8,
                reasoning=f"Extracted with interpretation bias",
                source_span="n=100 (approximately)"
            )
        return Extractor(extractor_id, extract_fn)
    
    extractors = [
        make_biased_extractor("extractor_1", bias=0),    # 100
        make_biased_extractor("extractor_2", bias=0),    # 100
        make_biased_extractor("extractor_3", bias=50),   # 150 (disagrees)
    ]
    
    critic = Critic("critic_1")
    arbitrator = Arbitrator("arbitrator_1")
    
    orchestrator = TCRivalsOrchestrator(
        extractors=extractors,
        critic=critic,
        arbitrator=arbitrator
    )
    
    task = ExtractionTask(
        task_id="disagreement_test",
        field_name="sample_size",
        source_context="The study enrolled approximately n=100 patients",
        extraction_type="numeric",
        validation_hints={"min": 10, "max": 500}
    )
    
    report = orchestrator.verify_task(task)
    
    print(f"\nField: {report.task.field_name}")
    print(f"Final Value: {report.final_value}")
    print(f"Status: {report.final_status}")
    print(f"Consensus: {report.consensus.consensus_type.name}")
    print(f"Agreement: {report.consensus.agreement_ratio:.0%}")
    
    if report.consensus.disagreement_details:
        print(f"Disagreement: {report.consensus.disagreement_details}")
    
    print("\nExtractor values:")
    for ext in report.extractions:
        if ext.value:
            print(f"  {ext.extractor_id}: {ext.value.value}")
    
    return report


# =============================================================================
# INTEGRATION WITH TRUTHCERT
# =============================================================================

def integrate_with_truthcert(report: VerificationReport) -> Dict[str, Any]:
    """
    Convert TC-RIVALS report to TruthCert-compatible format.
    
    TruthCert expects:
    - witness_count: Number of independent verifications
    - consensus_type: How agreement was reached
    - provenance: Audit trail
    - verification_status: Final gating decision
    """
    
    return {
        "truthcert_version": "1.0",
        "extension": "TC-RIVALS",
        "field": report.task.field_name,
        "value": report.final_value,
        "verification": {
            "status": report.final_status,
            "witness_count": len([e for e in report.extractions if e.status == ExtractionStatus.COMPLETE]),
            "consensus_type": report.consensus.consensus_type.name,
            "agreement_ratio": report.consensus.agreement_ratio,
            "confidence": report.consensus.confidence
        },
        "flags": [
            {
                "type": f.flag_type,
                "severity": f.severity.name,
                "description": f.description
            }
            for f in report.critic_flags
        ],
        "arbitration": {
            "required": report.arbitration is not None,
            "outcome": report.arbitration.outcome.name if report.arbitration else None,
            "human_review": report.arbitration.human_review_required if report.arbitration else False
        },
        "provenance": report.provenance,
        "audit_hash": hashlib.sha256(
            json.dumps({
                "task_id": report.task.task_id,
                "final_value": str(report.final_value),
                "status": report.final_status,
                "timestamp": report.provenance.get("timestamp", "")
            }, sort_keys=True).encode()
        ).hexdigest()[:16]
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run examples
    print("\n" + "#"*70)
    print("# TC-RIVALS: Team of Rivals Verification Extension")
    print("# TruthCert Protocol - Multi-Agent Verification")
    print("#"*70)
    
    # Example 1: RCT extraction
    reports = example_rct_extraction()
    
    # Example 2: Disagreement handling
    disagreement_report = example_with_disagreement()
    
    # Show TruthCert integration
    print("\n" + "="*70)
    print("TRUTHCERT INTEGRATION FORMAT")
    print("="*70)
    
    tc_format = integrate_with_truthcert(reports[0])
    print(json.dumps(tc_format, indent=2))
    
    print("\n" + "="*70)
    print("TC-RIVALS Implementation Complete")
    print("="*70)

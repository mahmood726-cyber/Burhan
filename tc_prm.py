"""
TC-PRM: Process Reward Model Verification Extension
====================================================

Based on:
- ReasonFlux-PRM (arXiv:2506.18896): Trajectory-aware process rewards
- SPARK (arXiv:2512.03244): Synthetic PRM training with anti-reward-hacking

Key Features:
- Step-level reward scoring (not just final answer)
- Trajectory-aware evaluation (considers full reasoning path)
- Anti-reward-hacking constraints
- Differentiates reasoning quality from answer correctness

TruthCert Integration:
- Scores each extraction/computation step
- Flags low-quality reasoning even if answer is correct
- Provides trajectory-level confidence scores
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json
import re
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TC-PRM")


# =============================================================================
# ENUMS
# =============================================================================

class StepType(Enum):
    """Types of reasoning steps."""
    EXTRACTION = auto()
    CALCULATION = auto()
    INFERENCE = auto()
    LOOKUP = auto()
    TRANSFORMATION = auto()
    AGGREGATION = auto()
    VALIDATION = auto()
    CONCLUSION = auto()


class StepQuality(Enum):
    """Quality assessment of a reasoning step."""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()
    INVALID = auto()


class TrajectoryQuality(Enum):
    """Overall quality of reasoning trajectory."""
    OPTIMAL = auto()
    GOOD = auto()
    SUBOPTIMAL = auto()
    FLAWED = auto()
    INVALID = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ReasoningStep:
    """A single step in a reasoning trajectory."""
    step_id: str
    step_number: int
    step_type: StepType
    content: str
    input_values: Dict[str, Any]
    output_value: Any
    justification: str
    source_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.step_id:
            self.step_id = hashlib.sha256(
                f"{self.step_number}:{self.content[:50]}".encode()
            ).hexdigest()[:12]


@dataclass
class StepReward:
    """Reward score for a single reasoning step."""
    step_id: str
    quality: StepQuality
    reward_score: float
    subscores: Dict[str, float]
    issues: List[str]
    suggestions: List[str]
    
    @property
    def is_acceptable(self) -> bool:
        return self.quality in [StepQuality.EXCELLENT, StepQuality.GOOD, StepQuality.ACCEPTABLE]


@dataclass
class Trajectory:
    """A complete reasoning trajectory."""
    trajectory_id: str
    task_description: str
    steps: List[ReasoningStep]
    final_answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.trajectory_id:
            self.trajectory_id = hashlib.sha256(
                f"{self.task_description}:{len(self.steps)}".encode()
            ).hexdigest()[:12]


@dataclass
class TrajectoryReward:
    """Reward score for an entire trajectory."""
    trajectory_id: str
    quality: TrajectoryQuality
    step_rewards: List[StepReward]
    trajectory_score: float
    answer_correctness: float
    reasoning_quality: float
    efficiency_score: float
    issues: List[str]
    reward_hacking_flags: List[str]
    
    @property
    def is_trustworthy(self) -> bool:
        return (
            self.quality in [TrajectoryQuality.OPTIMAL, TrajectoryQuality.GOOD] and
            len(self.reward_hacking_flags) == 0
        )


# =============================================================================
# STEP REWARD MODELS
# =============================================================================

class BaseStepRewardModel:
    """Base class for step-level reward models."""
    
    def score_step(self, step: ReasoningStep, context: Dict[str, Any]) -> StepReward:
        raise NotImplementedError


class ExtractionStepRewardModel(BaseStepRewardModel):
    """Reward model for extraction steps."""
    
    def score_step(self, step: ReasoningStep, context: Dict[str, Any]) -> StepReward:
        subscores = {}
        issues = []
        suggestions = []
        
        # Check source reference
        if step.source_reference:
            subscores["source_reference"] = 1.0
        else:
            subscores["source_reference"] = 0.0
            issues.append("No source reference provided")
            suggestions.append("Include specific source location")
        
        # Check justification quality
        if len(step.justification) > 20:
            subscores["justification"] = 0.8
        elif len(step.justification) > 5:
            subscores["justification"] = 0.5
            suggestions.append("Provide more detailed justification")
        else:
            subscores["justification"] = 0.2
            issues.append("Insufficient justification")
        
        # Check output value presence
        if step.output_value is not None:
            subscores["output_present"] = 1.0
        else:
            subscores["output_present"] = 0.0
            issues.append("No output value extracted")
        
        reward_score = sum(subscores.values()) / len(subscores) if subscores else 0
        
        if reward_score >= 0.9:
            quality = StepQuality.EXCELLENT
        elif reward_score >= 0.7:
            quality = StepQuality.GOOD
        elif reward_score >= 0.5:
            quality = StepQuality.ACCEPTABLE
        elif reward_score >= 0.3:
            quality = StepQuality.POOR
        else:
            quality = StepQuality.INVALID
        
        return StepReward(
            step_id=step.step_id,
            quality=quality,
            reward_score=reward_score * 2 - 1,
            subscores=subscores,
            issues=issues,
            suggestions=suggestions
        )


class CalculationStepRewardModel(BaseStepRewardModel):
    """Reward model for calculation steps."""
    
    def score_step(self, step: ReasoningStep, context: Dict[str, Any]) -> StepReward:
        subscores = {}
        issues = []
        suggestions = []
        
        # Check inputs
        if step.input_values:
            subscores["inputs_specified"] = 1.0
        else:
            subscores["inputs_specified"] = 0.3
            issues.append("Input values not specified")
            suggestions.append("List all input values used")
        
        # Check for formula in content
        formula_patterns = [r'=', r'\+', r'-', r'\*', r'/', r'\^']
        has_formula = any(re.search(p, step.content + step.justification) for p in formula_patterns)
        if has_formula:
            subscores["formula_present"] = 1.0
        else:
            subscores["formula_present"] = 0.4
            suggestions.append("Include explicit formula")
        
        # Check result plausibility
        if step.output_value is not None:
            try:
                val = float(step.output_value)
                if math.isfinite(val):
                    subscores["result_plausible"] = 1.0
                else:
                    subscores["result_plausible"] = 0.0
                    issues.append("Result is infinite or NaN")
            except (ValueError, TypeError):
                subscores["result_plausible"] = 0.5
        else:
            subscores["result_plausible"] = 0.0
            issues.append("No result produced")
        
        reward_score = sum(subscores.values()) / len(subscores) if subscores else 0
        
        if reward_score >= 0.9:
            quality = StepQuality.EXCELLENT
        elif reward_score >= 0.7:
            quality = StepQuality.GOOD
        elif reward_score >= 0.5:
            quality = StepQuality.ACCEPTABLE
        elif reward_score >= 0.3:
            quality = StepQuality.POOR
        else:
            quality = StepQuality.INVALID
        
        return StepReward(
            step_id=step.step_id,
            quality=quality,
            reward_score=reward_score * 2 - 1,
            subscores=subscores,
            issues=issues,
            suggestions=suggestions
        )


class InferenceStepRewardModel(BaseStepRewardModel):
    """Reward model for inference/reasoning steps."""
    
    def score_step(self, step: ReasoningStep, context: Dict[str, Any]) -> StepReward:
        subscores = {}
        issues = []
        suggestions = []
        
        # Check for premise indicators
        premise_indicators = ['because', 'since', 'given', 'as', 'from']
        has_premises = any(ind in step.content.lower() for ind in premise_indicators)
        if has_premises:
            subscores["premises_stated"] = 1.0
        else:
            subscores["premises_stated"] = 0.5
            suggestions.append("Explicitly state premises")
        
        # Check for conclusion indicators
        conclusion_indicators = ['therefore', 'thus', 'hence', 'so', 'conclude']
        has_conclusion = any(ind in step.content.lower() for ind in conclusion_indicators)
        if has_conclusion:
            subscores["conclusion_clear"] = 1.0
        else:
            subscores["conclusion_clear"] = 0.6
            suggestions.append("Use explicit conclusion markers")
        
        # Check justification length
        if len(step.justification) > 50:
            subscores["reasoning_depth"] = 1.0
        elif len(step.justification) > 20:
            subscores["reasoning_depth"] = 0.7
        else:
            subscores["reasoning_depth"] = 0.4
            suggestions.append("Provide more detailed reasoning")
        
        reward_score = sum(subscores.values()) / len(subscores) if subscores else 0
        
        if reward_score >= 0.85:
            quality = StepQuality.EXCELLENT
        elif reward_score >= 0.7:
            quality = StepQuality.GOOD
        elif reward_score >= 0.5:
            quality = StepQuality.ACCEPTABLE
        elif reward_score >= 0.3:
            quality = StepQuality.POOR
        else:
            quality = StepQuality.INVALID
        
        return StepReward(
            step_id=step.step_id,
            quality=quality,
            reward_score=reward_score * 2 - 1,
            subscores=subscores,
            issues=issues,
            suggestions=suggestions
        )


# =============================================================================
# ANTI-REWARD-HACKING
# =============================================================================

class AntiRewardHackingDetector:
    """Detect potential reward hacking behaviors (based on SPARK)."""
    
    def detect(self, trajectory: Trajectory, step_rewards: List[StepReward]) -> List[str]:
        flags = []
        
        # Check 1: Verbosity hacking
        avg_length = sum(len(s.content) for s in trajectory.steps) / len(trajectory.steps) if trajectory.steps else 0
        if avg_length > 500:
            flags.append("VERBOSITY: Average step length unusually high")
        
        # Check 2: Repetition gaming
        contents = [s.content.lower() for s in trajectory.steps]
        unique_ratio = len(set(contents)) / len(contents) if contents else 1
        if unique_ratio < 0.7:
            flags.append("REPETITION: Steps contain repeated content")
        
        # Check 3: Keyword stuffing
        keyword_indicators = ['therefore', 'thus', 'because', 'calculated', 'verified']
        total_text = ' '.join(s.content + s.justification for s in trajectory.steps).lower()
        word_count = len(total_text.split()) if total_text else 1
        keyword_density = sum(total_text.count(kw) for kw in keyword_indicators) / word_count
        if keyword_density > 0.1:
            flags.append("KEYWORD_STUFFING: Excessive reasoning keywords")
        
        # Check 4: Suspiciously uniform rewards
        if step_rewards and len(step_rewards) > 3:
            scores = [r.reward_score for r in step_rewards]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean)**2 for s in scores) / len(scores)
            if variance < 0.01:
                flags.append("UNIFORM_SCORES: Step scores suspiciously uniform")
        
        # Check 5: Short-circuit detection
        if len(trajectory.steps) < 2 and trajectory.metadata.get('confidence', 0) > 0.9:
            flags.append("SHORT_CIRCUIT: Very short trajectory with high confidence")
        
        return flags


# =============================================================================
# TRAJECTORY REWARD MODEL
# =============================================================================

class TCPRM:
    """Main TC-PRM class combining step-level and trajectory-level evaluation."""
    
    def __init__(self):
        self.step_models: Dict[StepType, BaseStepRewardModel] = {
            StepType.EXTRACTION: ExtractionStepRewardModel(),
            StepType.CALCULATION: CalculationStepRewardModel(),
            StepType.INFERENCE: InferenceStepRewardModel(),
        }
        self.anti_hacking = AntiRewardHackingDetector()
        self.evaluation_count = 0
    
    def _get_step_model(self, step_type: StepType) -> BaseStepRewardModel:
        if step_type in self.step_models:
            return self.step_models[step_type]
        return self.step_models[StepType.INFERENCE]
    
    def score_step(self, step: ReasoningStep, context: Dict[str, Any] = None) -> StepReward:
        context = context or {}
        model = self._get_step_model(step.step_type)
        return model.score_step(step, context)
    
    def score_trajectory(
        self, 
        trajectory: Trajectory,
        expected_answer: Optional[Any] = None,
        context: Dict[str, Any] = None
    ) -> TrajectoryReward:
        self.evaluation_count += 1
        context = context or {}
        
        # Score each step
        step_rewards = [self.score_step(step, context) for step in trajectory.steps]
        
        # Calculate trajectory-level metrics
        if step_rewards:
            reasoning_quality = (sum(r.reward_score for r in step_rewards) / len(step_rewards) + 1) / 2
        else:
            reasoning_quality = 0.0
        
        # Answer correctness
        if expected_answer is not None:
            try:
                if str(trajectory.final_answer).strip() == str(expected_answer).strip():
                    answer_correctness = 1.0
                elif isinstance(trajectory.final_answer, (int, float)) and isinstance(expected_answer, (int, float)):
                    if abs(float(trajectory.final_answer) - float(expected_answer)) < 0.01:
                        answer_correctness = 1.0
                    else:
                        answer_correctness = 0.0
                else:
                    answer_correctness = 0.0
            except:
                answer_correctness = 0.5
        else:
            answer_correctness = 0.5
        
        # Efficiency score
        optimal_steps = 3
        actual_steps = len(trajectory.steps)
        if actual_steps == 0:
            efficiency_score = 0.0
        elif actual_steps <= optimal_steps:
            efficiency_score = 1.0
        else:
            efficiency_score = max(0.3, optimal_steps / actual_steps)
        
        # Check for reward hacking
        hacking_flags = self.anti_hacking.detect(trajectory, step_rewards)
        hacking_penalty = 0.2 * len(hacking_flags)
        
        # Calculate overall trajectory score
        trajectory_score = (
            0.5 * reasoning_quality +
            0.3 * answer_correctness +
            0.2 * efficiency_score -
            hacking_penalty
        )
        trajectory_score = max(0.0, min(1.0, trajectory_score))
        
        # Collect issues
        issues = []
        for reward in step_rewards:
            issues.extend(reward.issues)
        
        bad_steps = sum(1 for r in step_rewards if not r.is_acceptable)
        if bad_steps > 0:
            issues.append(f"{bad_steps}/{len(step_rewards)} steps have quality issues")
        
        if efficiency_score < 0.5:
            issues.append(f"Trajectory may be inefficient ({actual_steps} steps)")
        
        # Determine quality level
        if trajectory_score >= 0.85 and len(hacking_flags) == 0:
            quality = TrajectoryQuality.OPTIMAL
        elif trajectory_score >= 0.7 and len(hacking_flags) == 0:
            quality = TrajectoryQuality.GOOD
        elif trajectory_score >= 0.5:
            quality = TrajectoryQuality.SUBOPTIMAL
        elif trajectory_score >= 0.3:
            quality = TrajectoryQuality.FLAWED
        else:
            quality = TrajectoryQuality.INVALID
        
        return TrajectoryReward(
            trajectory_id=trajectory.trajectory_id,
            quality=quality,
            step_rewards=step_rewards,
            trajectory_score=trajectory_score,
            answer_correctness=answer_correctness,
            reasoning_quality=reasoning_quality,
            efficiency_score=efficiency_score,
            issues=issues,
            reward_hacking_flags=hacking_flags
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.evaluation_count,
            "step_models": list(self.step_models.keys())
        }


# =============================================================================
# META-ANALYSIS EXAMPLE
# =============================================================================

def example_meta_analysis_trajectory():
    """Example: Score a meta-analysis data extraction trajectory."""
    print("\n" + "="*70)
    print("TC-PRM EXAMPLE: Meta-Analysis Extraction Trajectory")
    print("="*70)
    
    prm = TCPRM()
    
    trajectory = Trajectory(
        trajectory_id="",
        task_description="Extract relative risk from COLCOT trial",
        steps=[
            ReasoningStep(
                step_id="", step_number=1, step_type=StepType.EXTRACTION,
                content="Extract treatment group sample size",
                input_values={}, output_value=156,
                justification="Found in Methods: 'randomized to colchicine (n=156)'",
                source_reference="COLCOT trial, Methods, paragraph 2"
            ),
            ReasoningStep(
                step_id="", step_number=2, step_type=StepType.EXTRACTION,
                content="Extract control group sample size",
                input_values={}, output_value=151,
                justification="Found in Methods: 'placebo (n=151)'",
                source_reference="COLCOT trial, Methods, paragraph 2"
            ),
            ReasoningStep(
                step_id="", step_number=3, step_type=StepType.EXTRACTION,
                content="Extract treatment group events",
                input_values={}, output_value=23,
                justification="Results: '23 patients (14.7%) in colchicine group'",
                source_reference="COLCOT trial, Results, Table 2"
            ),
            ReasoningStep(
                step_id="", step_number=4, step_type=StepType.EXTRACTION,
                content="Extract control group events",
                input_values={}, output_value=40,
                justification="Results: '40 patients (26.5%) in placebo group'",
                source_reference="COLCOT trial, Results, Table 2"
            ),
            ReasoningStep(
                step_id="", step_number=5, step_type=StepType.CALCULATION,
                content="Calculate relative risk: RR = (a/n1) / (b/n2)",
                input_values={"events_treatment": 23, "n_treatment": 156, "events_control": 40, "n_control": 151},
                output_value=0.556,
                justification="RR = (23/156) / (40/151) = 0.1474 / 0.2649 = 0.556"
            ),
            ReasoningStep(
                step_id="", step_number=6, step_type=StepType.INFERENCE,
                content="Interpret the relative risk result",
                input_values={"RR": 0.556}, output_value="Protective effect",
                justification="Because RR < 1 (0.556), therefore colchicine shows protective effect, reducing MACE risk by ~44%"
            ),
        ],
        final_answer=0.556
    )
    
    result = prm.score_trajectory(trajectory, expected_answer=0.556)
    
    print(f"\nTrajectory: {trajectory.task_description}")
    print(f"Final Answer: {trajectory.final_answer}")
    print(f"\nOverall Quality: {result.quality.name}")
    print(f"Trajectory Score: {result.trajectory_score:.3f}")
    print(f"  - Reasoning Quality: {result.reasoning_quality:.3f}")
    print(f"  - Answer Correctness: {result.answer_correctness:.3f}")
    print(f"  - Efficiency Score: {result.efficiency_score:.3f}")
    
    print("\n" + "-"*70)
    print("STEP-BY-STEP REWARDS")
    print("-"*70)
    
    for step, reward in zip(trajectory.steps, result.step_rewards):
        print(f"\nStep {step.step_number}: {step.content[:50]}...")
        print(f"  Type: {step.step_type.name}, Quality: {reward.quality.name}, Score: {reward.reward_score:.3f}")
        if reward.issues:
            print(f"  Issues: {reward.issues}")
    
    print(f"\nTrajectory Trustworthy: {result.is_trustworthy}")
    return result


def example_poor_trajectory():
    """Example: Score a poor-quality trajectory."""
    print("\n" + "="*70)
    print("TC-PRM EXAMPLE: Poor Quality Trajectory Detection")
    print("="*70)
    
    prm = TCPRM()
    
    trajectory = Trajectory(
        trajectory_id="",
        task_description="Extract data from trial",
        steps=[
            ReasoningStep(
                step_id="", step_number=1, step_type=StepType.EXTRACTION,
                content="Got number", input_values={}, output_value=100,
                justification="It's there"
            ),
            ReasoningStep(
                step_id="", step_number=2, step_type=StepType.CALCULATION,
                content="Did calculation", input_values={}, output_value=50,
                justification="Math"
            ),
        ],
        final_answer=50,
        metadata={"confidence": 0.95}
    )
    
    result = prm.score_trajectory(trajectory, expected_answer=75)
    
    print(f"\nTrajectory Quality: {result.quality.name}")
    print(f"Trajectory Score: {result.trajectory_score:.3f}")
    print(f"Answer Correctness: {result.answer_correctness:.3f}")
    
    print("\nIssues Detected:")
    for issue in result.issues:
        print(f"  - {issue}")
    
    print("\nReward Hacking Flags:")
    for flag in result.reward_hacking_flags:
        print(f"  ⚠️ {flag}")
    
    print(f"\nTrajectory Trustworthy: {result.is_trustworthy}")
    return result


# =============================================================================
# INTEGRATION WITH TRUTHCERT
# =============================================================================

def integrate_with_truthcert(result: TrajectoryReward, trajectory: Trajectory) -> Dict[str, Any]:
    """Convert TC-PRM result to TruthCert-compatible format."""
    if result.is_trustworthy:
        status = "verified"
    elif result.quality == TrajectoryQuality.SUBOPTIMAL:
        status = "flagged"
    else:
        status = "rejected"
    
    return {
        "truthcert_version": "1.0",
        "extension": "TC-PRM",
        "trajectory": {
            "id": trajectory.trajectory_id,
            "task": trajectory.task_description,
            "steps": len(trajectory.steps),
            "final_answer": trajectory.final_answer
        },
        "verification": {
            "status": status,
            "quality": result.quality.name,
            "trajectory_score": result.trajectory_score,
            "trustworthy": result.is_trustworthy
        },
        "scores": {
            "reasoning_quality": result.reasoning_quality,
            "answer_correctness": result.answer_correctness,
            "efficiency": result.efficiency_score
        },
        "step_analysis": [
            {"step_id": r.step_id, "quality": r.quality.name, "score": r.reward_score}
            for r in result.step_rewards
        ],
        "flags": {
            "issues": result.issues,
            "reward_hacking": result.reward_hacking_flags
        },
        "audit_hash": hashlib.sha256(
            json.dumps({"id": trajectory.trajectory_id, "score": result.trajectory_score}, sort_keys=True).encode()
        ).hexdigest()[:16]
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# TC-PRM: Process Reward Model Verification Extension")
    print("# TruthCert Protocol - Step-Level Reasoning Verification")
    print("#"*70)
    
    good_result = example_meta_analysis_trajectory()
    poor_result = example_poor_trajectory()
    
    print("\n" + "="*70)
    print("TRUTHCERT INTEGRATION FORMAT")
    print("="*70)
    
    sample_trajectory = Trajectory(
        trajectory_id="demo_001",
        task_description="Demo extraction",
        steps=[],
        final_answer=0.556
    )
    
    tc_format = integrate_with_truthcert(good_result, sample_trajectory)
    print(json.dumps(tc_format, indent=2))
    
    print("\n" + "="*70)
    print("TC-PRM Implementation Complete")
    print("="*70)

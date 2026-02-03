"""
TC-QWED: Deterministic Verification Engines Extension for TruthCert
====================================================================

Implements the QWED protocol for deterministic verification of LLM outputs
using symbolic engines (SymPy, Z3, AST analysis). Based on QWED-AI/qwed-verification.

Core principle: Treat the LLM as an untrusted translator; trust it only to
translate natural language to formal DSL. Then verify with deterministic engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Tuple
import re
import ast
import json


class EngineType(Enum):
    """Types of deterministic verification engines."""
    MATH = "math"           # SymPy for mathematical verification
    LOGIC = "logic"         # Z3 for logical constraints
    STATS = "stats"         # Statistical computations
    FACT = "fact"           # Knowledge base lookup
    CODE = "code"           # AST analysis, linting
    SQL = "sql"             # SQL query validation
    IMAGE = "image"         # Visual content verification
    REASONING = "reasoning" # Logical inference chains


class VerificationStatus(Enum):
    """Status of verification."""
    VERIFIED = "verified"       # Deterministically proven correct
    CORRECTED = "corrected"     # Wrong, but correction provided
    REJECTED = "rejected"       # Failed verification, no correction
    UNSUPPORTED = "unsupported" # Engine cannot handle this input


@dataclass
class VerificationResult:
    """Result from a deterministic verification engine."""
    status: VerificationStatus
    engine: EngineType
    input_claim: str
    verified_value: Optional[Any] = None
    computed_value: Optional[Any] = None
    correction: Optional[str] = None
    proof: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class VerificationEngine(ABC):
    """Abstract base class for verification engines."""
    
    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        """Return the type of this engine."""
        pass
    
    @abstractmethod
    def verify(self, claim: str, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Verify a claim and return result."""
        pass
    
    @abstractmethod
    def can_handle(self, claim: str) -> bool:
        """Check if this engine can handle the given claim."""
        pass


class MathEngine(VerificationEngine):
    """
    Mathematical verification using SymPy.
    
    Handles:
    - Arithmetic expressions
    - Algebraic equations
    - Calculus (derivatives, integrals)
    - Effect size calculations
    """
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.MATH
    
    def can_handle(self, claim: str) -> bool:
        """Check for mathematical patterns."""
        math_patterns = [
            r'[+\-*/^]',           # Basic operators
            r'=',                   # Equations
            r'\d+\.\d+',           # Decimals
            r'sqrt|log|exp|ln',    # Functions
            r'RR|OR|HR|MD|SMD',    # Effect size indicators
        ]
        return any(re.search(p, claim, re.IGNORECASE) for p in math_patterns)
    
    def verify(self, claim: str, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Verify mathematical claim using SymPy."""
        try:
            # Dynamic import to avoid hard dependency
            from sympy import sympify, simplify, Eq, solve, N
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
            
            context = context or {}
            
            # Handle equation verification
            if '=' in claim and not claim.startswith('='):
                parts = claim.split('=')
                if len(parts) == 2:
                    lhs = self._parse_expr(parts[0].strip(), context)
                    rhs = self._parse_expr(parts[1].strip(), context)
                    
                    # Check if equation holds
                    diff = simplify(lhs - rhs)
                    
                    if diff == 0:
                        return VerificationResult(
                            status=VerificationStatus.VERIFIED,
                            engine=self.engine_type,
                            input_claim=claim,
                            verified_value=True,
                            proof=f"LHS = RHS = {N(lhs, 6)}"
                        )
                    else:
                        computed_rhs = N(lhs, 6)
                        return VerificationResult(
                            status=VerificationStatus.CORRECTED,
                            engine=self.engine_type,
                            input_claim=claim,
                            verified_value=False,
                            computed_value=float(computed_rhs),
                            correction=f"The correct answer is {computed_rhs}, not {parts[1].strip()}"
                        )
            
            # Handle expression evaluation
            else:
                expr = self._parse_expr(claim, context)
                result = N(expr, 6)
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    engine=self.engine_type,
                    input_claim=claim,
                    computed_value=float(result)
                )
                
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                engine=self.engine_type,
                input_claim=claim,
                error_message=str(e)
            )
    
    def _parse_expr(self, expr_str: str, context: Dict[str, Any]):
        """Parse expression with context substitution."""
        from sympy import sympify, Symbol
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
        
        # Replace context variables
        for var, val in context.items():
            expr_str = re.sub(rf'\b{var}\b', str(val), expr_str)
        
        # Parse with transformations
        transformations = standard_transformations + (implicit_multiplication,)
        return parse_expr(expr_str, transformations=transformations)


class LogicEngine(VerificationEngine):
    """
    Logical verification using Z3.
    
    Handles:
    - Propositional logic
    - First-order logic constraints
    - Satisfiability checking
    - Eligibility criteria verification
    """
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.LOGIC
    
    def can_handle(self, claim: str) -> bool:
        """Check for logical patterns."""
        logic_patterns = [
            r'\bAND\b|\bOR\b|\bNOT\b',
            r'\bIF\b.*\bTHEN\b',
            r'\bIMPLIES\b',
            r'\bFORALL\b|\bEXISTS\b',
            r'\bGT\b|\bLT\b|\bGE\b|\bLE\b|\bEQ\b',
            r'=>|&&|\|\|',
        ]
        return any(re.search(p, claim, re.IGNORECASE) for p in logic_patterns)
    
    def verify(self, claim: str, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Verify logical claim using Z3."""
        try:
            from z3 import Solver, Bool, Int, Real, And, Or, Not, Implies, sat, unsat
            
            context = context or {}
            solver = Solver()
            
            # Parse LISP-style DSL: (AND (GT x 5) (LT y 10))
            constraint = self._parse_logic_dsl(claim, context)
            solver.add(constraint)
            
            result = solver.check()
            
            if result == sat:
                model = solver.model()
                model_dict = {str(d): model[d] for d in model}
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    engine=self.engine_type,
                    input_claim=claim,
                    verified_value=True,
                    computed_value=model_dict,
                    proof=f"Satisfiable with model: {model_dict}"
                )
            elif result == unsat:
                return VerificationResult(
                    status=VerificationStatus.CORRECTED,
                    engine=self.engine_type,
                    input_claim=claim,
                    verified_value=False,
                    correction="Constraint is unsatisfiable - no valid assignment exists"
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.UNSUPPORTED,
                    engine=self.engine_type,
                    input_claim=claim,
                    error_message="Z3 returned unknown"
                )
                
        except ImportError:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                engine=self.engine_type,
                input_claim=claim,
                error_message="Z3 not installed. Install with: pip install z3-solver"
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                engine=self.engine_type,
                input_claim=claim,
                error_message=str(e)
            )
    
    def _parse_logic_dsl(self, dsl: str, context: Dict[str, Any]):
        """Parse LISP-style logic DSL to Z3 constraints."""
        from z3 import Int, Real, And, Or, Not, Implies
        
        # Tokenize
        tokens = dsl.replace('(', ' ( ').replace(')', ' ) ').split()
        
        variables = {}
        
        def get_var(name: str):
            if name in context:
                return context[name]
            if name not in variables:
                # Default to Int, could be smarter about type inference
                variables[name] = Int(name)
            return variables[name]
        
        def parse_expr(tokens: List[str], idx: int) -> Tuple[Any, int]:
            if tokens[idx] != '(':
                # Atom
                token = tokens[idx]
                try:
                    return int(token), idx + 1
                except ValueError:
                    try:
                        return float(token), idx + 1
                    except ValueError:
                        return get_var(token), idx + 1
            
            # List expression
            idx += 1  # Skip '('
            op = tokens[idx].upper()
            idx += 1
            
            args = []
            while tokens[idx] != ')':
                arg, idx = parse_expr(tokens, idx)
                args.append(arg)
            idx += 1  # Skip ')'
            
            # Build Z3 expression
            if op == 'AND':
                return And(*args), idx
            elif op == 'OR':
                return Or(*args), idx
            elif op == 'NOT':
                return Not(args[0]), idx
            elif op == 'IMPLIES':
                return Implies(args[0], args[1]), idx
            elif op == 'GT':
                return args[0] > args[1], idx
            elif op == 'LT':
                return args[0] < args[1], idx
            elif op == 'GE':
                return args[0] >= args[1], idx
            elif op == 'LE':
                return args[0] <= args[1], idx
            elif op == 'EQ':
                return args[0] == args[1], idx
            else:
                raise ValueError(f"Unknown operator: {op}")
        
        result, _ = parse_expr(tokens, 0)
        return result


class StatsEngine(VerificationEngine):
    """
    Statistical verification.
    
    Handles:
    - Heterogeneity calculations (I², Q, tau²)
    - Effect size conversions
    - Confidence interval calculations
    - Meta-analysis statistics
    """
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.STATS
    
    def can_handle(self, claim: str) -> bool:
        """Check for statistical patterns."""
        stats_patterns = [
            r'\bI2\b|\bI\^2\b|I²',
            r'\btau2\b|\bτ²',
            r'\bQ\b.*statistic',
            r'\bCI\b|\bconfidence interval\b',
            r'\bSE\b|\bstandard error\b',
            r'\bp-value\b|\bp\s*[<=]',
        ]
        return any(re.search(p, claim, re.IGNORECASE) for p in stats_patterns)
    
    def verify(self, claim: str, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Verify statistical claim."""
        context = context or {}
        
        try:
            # I² calculation: I² = max(0, (Q - df) / Q * 100)
            if re.search(r'I2|I\^2|I²', claim, re.IGNORECASE):
                if 'Q' in context and 'df' in context:
                    Q = context['Q']
                    df = context['df']
                    computed_I2 = max(0, (Q - df) / Q * 100)
                    
                    # Extract claimed value
                    match = re.search(r'=\s*([\d.]+)', claim)
                    if match:
                        claimed = float(match.group(1))
                        if abs(computed_I2 - claimed) < 0.5:  # Allow 0.5% tolerance
                            return VerificationResult(
                                status=VerificationStatus.VERIFIED,
                                engine=self.engine_type,
                                input_claim=claim,
                                verified_value=True,
                                computed_value=round(computed_I2, 2)
                            )
                        else:
                            return VerificationResult(
                                status=VerificationStatus.CORRECTED,
                                engine=self.engine_type,
                                input_claim=claim,
                                verified_value=False,
                                computed_value=round(computed_I2, 2),
                                correction=f"I² should be {computed_I2:.2f}%, not {claimed}%"
                            )
                    
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        engine=self.engine_type,
                        input_claim=claim,
                        computed_value=round(computed_I2, 2)
                    )
            
            # SE from CI: SE = (upper - lower) / (2 * 1.96)
            if 'upper' in context and 'lower' in context:
                upper = context['upper']
                lower = context['lower']
                computed_SE = (upper - lower) / (2 * 1.96)
                
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    engine=self.engine_type,
                    input_claim=claim,
                    computed_value=round(computed_SE, 4)
                )
            
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                engine=self.engine_type,
                input_claim=claim,
                error_message="Insufficient context for statistical verification"
            )
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                engine=self.engine_type,
                input_claim=claim,
                error_message=str(e)
            )


class CodeEngine(VerificationEngine):
    """
    Code verification using AST analysis.
    
    Handles:
    - Python syntax validation
    - Security vulnerability detection
    - Code style checking
    """
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.CODE
    
    def can_handle(self, claim: str) -> bool:
        """Check for code patterns."""
        code_patterns = [
            r'def\s+\w+\s*\(',
            r'import\s+\w+',
            r'class\s+\w+',
            r'for\s+\w+\s+in\s+',
            r'if\s+.*:',
        ]
        return any(re.search(p, claim) for p in code_patterns)
    
    def verify(self, claim: str, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Verify code using AST analysis."""
        context = context or {}
        language = context.get('language', 'python')
        
        if language != 'python':
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                engine=self.engine_type,
                input_claim=claim[:100] + "...",
                error_message=f"Language '{language}' not supported, only Python"
            )
        
        try:
            # Parse AST
            tree = ast.parse(claim)
            
            # Check for common security issues
            vulnerabilities = self._check_security(tree)
            
            if vulnerabilities:
                return VerificationResult(
                    status=VerificationStatus.CORRECTED,
                    engine=self.engine_type,
                    input_claim=claim[:100] + "...",
                    verified_value=False,
                    correction=f"Security issues found: {', '.join(vulnerabilities)}"
                )
            
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                engine=self.engine_type,
                input_claim=claim[:100] + "...",
                verified_value=True,
                proof="AST parsed successfully, no security issues detected"
            )
            
        except SyntaxError as e:
            return VerificationResult(
                status=VerificationStatus.REJECTED,
                engine=self.engine_type,
                input_claim=claim[:100] + "...",
                error_message=f"Syntax error at line {e.lineno}: {e.msg}"
            )
    
    def _check_security(self, tree: ast.AST) -> List[str]:
        """Check for common security vulnerabilities."""
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for eval/exec
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec'):
                        vulnerabilities.append(f"Dangerous function: {node.func.id}")
            
            # Check for shell=True in subprocess
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            vulnerabilities.append("subprocess with shell=True")
        
        return vulnerabilities


class TCQWEDValidator:
    """
    Main validator integrating all deterministic verification engines.
    
    Implements the QWED protocol: LLM translates, engines verify.
    """
    
    def __init__(self):
        self.engines: List[VerificationEngine] = [
            MathEngine(),
            LogicEngine(),
            StatsEngine(),
            CodeEngine(),
        ]
    
    def add_engine(self, engine: VerificationEngine) -> None:
        """Add a custom verification engine."""
        self.engines.append(engine)
    
    def verify(
        self,
        claim: str,
        context: Optional[Dict[str, Any]] = None,
        engine_type: Optional[EngineType] = None
    ) -> VerificationResult:
        """
        Verify a claim using appropriate engine.
        
        Args:
            claim: The claim to verify
            context: Additional context (variable values, etc.)
            engine_type: Force specific engine (auto-detect if None)
            
        Returns:
            VerificationResult from the engine
        """
        # Find appropriate engine
        if engine_type:
            engines = [e for e in self.engines if e.engine_type == engine_type]
        else:
            engines = [e for e in self.engines if e.can_handle(claim)]
        
        if not engines:
            return VerificationResult(
                status=VerificationStatus.UNSUPPORTED,
                engine=EngineType.MATH,  # Default
                input_claim=claim,
                error_message="No suitable verification engine found"
            )
        
        # Use first matching engine
        return engines[0].verify(claim, context)
    
    def verify_extraction(
        self,
        extraction: Dict[str, Any],
        formulas: Optional[Dict[str, str]] = None,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, VerificationResult]:
        """
        Verify all verifiable components of an extraction.
        
        Args:
            extraction: Field name -> value mapping
            formulas: Field name -> formula string (for derived values)
            constraints: List of logical constraints to verify
            
        Returns:
            Dictionary of field/constraint -> VerificationResult
        """
        results = {}
        formulas = formulas or {}
        constraints = constraints or []
        
        # Verify formulas
        for field, formula in formulas.items():
            if field in extraction:
                full_claim = f"{formula} = {extraction[field]}"
                results[f"formula:{field}"] = self.verify(full_claim, extraction)
        
        # Verify constraints
        for i, constraint in enumerate(constraints):
            results[f"constraint:{i}"] = self.verify(
                constraint,
                extraction,
                engine_type=EngineType.LOGIC
            )
        
        return results


# Example usage
def example_meta_analysis_verification():
    """Demonstrate TC-QWED for meta-analysis calculations."""
    
    validator = TCQWEDValidator()
    
    print("TC-QWED Verification Examples")
    print("=" * 50)
    
    # 1. Math verification: Effect size calculation
    print("\n1. Effect Size Calculation:")
    result = validator.verify(
        "RR = 23/156 / (40/151)",
        context={}
    )
    print(f"   Claim: RR = 23/156 / (40/151)")
    print(f"   Status: {result.status.value}")
    print(f"   Computed: {result.computed_value}")
    
    # 2. Math verification with error
    print("\n2. Incorrect Calculation:")
    result = validator.verify(
        "0.73 * 0.58 = 0.50",
        context={}
    )
    print(f"   Claim: 0.73 * 0.58 = 0.50")
    print(f"   Status: {result.status.value}")
    print(f"   Correction: {result.correction}")
    
    # 3. Logic verification: Eligibility criteria
    print("\n3. Eligibility Criteria Logic:")
    result = validator.verify(
        "(AND (GE age 18) (LE age 80) (NOT pregnant))",
        context={"age": 45, "pregnant": False}
    )
    print(f"   Constraint: (AND (GE age 18) (LE age 80) (NOT pregnant))")
    print(f"   Status: {result.status.value}")
    print(f"   Model: {result.computed_value}")
    
    # 4. Statistics verification: I² calculation
    print("\n4. Heterogeneity (I²) Calculation:")
    result = validator.verify(
        "I² = 73.45",
        context={"Q": 45.2, "df": 12}
    )
    print(f"   Claim: I² = 73.45")
    print(f"   Status: {result.status.value}")
    print(f"   Computed: {result.computed_value}")
    
    # 5. Code verification
    print("\n5. Code Security Check:")
    code = """
def calculate_effect_size(events_i, total_i, events_c, total_c):
    rr = (events_i / total_i) / (events_c / total_c)
    return rr
"""
    result = validator.verify(code, context={"language": "python"})
    print(f"   Status: {result.status.value}")
    print(f"   Proof: {result.proof}")
    
    # 6. Full extraction verification
    print("\n6. Full Extraction Verification:")
    extraction = {
        "events_intervention": 23,
        "total_intervention": 156,
        "events_control": 40,
        "total_control": 151,
        "risk_ratio": 0.556,
        "Q": 45.2,
        "df": 12,
        "I2": 73.45
    }
    
    formulas = {
        "risk_ratio": "(events_intervention/total_intervention)/(events_control/total_control)"
    }
    
    constraints = [
        "(AND (GT events_intervention 0) (LE events_intervention total_intervention))",
        "(AND (GT events_control 0) (LE events_control total_control))"
    ]
    
    results = validator.verify_extraction(extraction, formulas, constraints)
    
    for key, result in results.items():
        print(f"   {key}: {result.status.value}")


if __name__ == "__main__":
    example_meta_analysis_verification()

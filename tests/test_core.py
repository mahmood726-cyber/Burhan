"""Core regression tests for the TruthCert verification modules and benchmark scripts.

These tests exercise the import-level and functional paths that the original
smoke test never touched. In particular:

- Importing ``simulate_truthcert_vs_rct_pack`` runs module-level code that loads
  its sibling ``truthcert_toy_benchmark.py`` by relative path. A wrong relative
  path (regression F1) made that import raise ``FileNotFoundError`` on every run.
- ``score_contract_v1.score_file`` must return ``None`` for an empty JSONL file
  rather than dividing by zero.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TC_MODULES = [
    "tc_pcn",
    "tc_prm",
    "tc_qwed",
    "tc_formal",
    "tc_rivals",
    "tc_verifier",
    "tc_integration",
    "score_contract_v1",
    "truthcert_toy_benchmark",
]


@pytest.mark.parametrize("module_name", TC_MODULES)
def test_module_imports(module_name):
    """Every core module must import without raising."""
    assert importlib.import_module(module_name) is not None


def test_simulate_import_and_summarize():
    """Regression for F1: the sibling module must load and the sim must run.

    Before the path fix, importing this module raised FileNotFoundError because
    it looked for truthcert_toy_benchmark.py one directory too high.
    """
    sim = importlib.import_module("simulate_truthcert_vs_rct_pack")
    df = sim.summarize(n=5, seed=1)
    # summarize returns one row per protocol.
    assert len(df) == 2
    assert set(df["protocol"]) == {"TruthCert", "TruthCert+TC-RCT"}


def test_score_file_empty_returns_none(tmp_path):
    """Regression for F4: score_file must not divide by zero on empty input."""
    score = importlib.import_module("score_contract_v1")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert score.score_file(empty) is None


def test_score_file_basic(tmp_path):
    """score_file computes the expected fractions on a tiny hand-built log."""
    score = importlib.import_module("score_contract_v1")
    log = tmp_path / "runs.jsonl"
    log.write_text(
        "\n".join(
            [
                '{"shipped": true, "rejected": false, "bundle_correct": true, "tokens": 100}',
                '{"shipped": true, "rejected": false, "bundle_correct": false, "tokens": 100}',
                '{"shipped": false, "rejected": true, "bundle_correct": false, "tokens": 100}',
            ]
        ),
        encoding="utf-8",
    )
    s = score.score_file(log)
    assert s["n_scenarios"] == 3
    assert s["shipped_pct"] == pytest.approx(2 / 3)
    assert s["false_ship_pct"] == pytest.approx(1 / 3)
    assert s["reject_pct"] == pytest.approx(1 / 3)
    assert s["tokens_per_correct_shipped"] == pytest.approx(300.0)

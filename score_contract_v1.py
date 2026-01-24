#!/usr/bin/env python3
"""score_contract_v1.py

Compute Contract-v1 metrics from JSONL run logs.
Expected structure:
  <bench>/runs/<protocol>/*.jsonl

Each JSON object must include:
  shipped: bool
  rejected: bool
  bundle_correct: bool
  tokens: int

Usage:
  python score_contract_v1.py <bench_folder>
"""
import json, sys
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def score_file(path: Path):
    rows = list(read_jsonl(path))
    n = len(rows)
    if n == 0:
        return None
    shipped = sum(1 for r in rows if r.get("shipped") is True)
    rejected = sum(1 for r in rows if r.get("rejected") is True)
    false_ship = sum(1 for r in rows if (r.get("shipped") is True and r.get("bundle_correct") is False))
    tokens_total = sum(int(r.get("tokens", 0)) for r in rows)
    correct_shipped = sum(1 for r in rows if (r.get("shipped") is True and r.get("bundle_correct") is True))
    return {
        "n_scenarios": n,
        "shipped_pct": shipped / n,
        "false_ship_pct": false_ship / n,
        "reject_pct": rejected / n,
        "mean_tokens_per_bundle": tokens_total / n,
        "tokens_per_correct_shipped": (tokens_total / correct_shipped) if correct_shipped else float("inf"),
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python score_contract_v1.py <bench_folder>", file=sys.stderr)
        raise SystemExit(2)

    bench = Path(sys.argv[1]).resolve()
    runs = bench / "runs"
    if not runs.exists():
        print(f"Missing runs/ in {bench}", file=sys.stderr)
        raise SystemExit(2)

    print("protocol\tfile\tn_scenarios\tshipped_pct\tfalse_ship_pct\treject_pct\tmean_tokens_per_bundle\ttokens_per_correct_shipped")
    any_found = False
    for prot in sorted([p for p in runs.iterdir() if p.is_dir()]):
        for f in sorted(prot.glob("*.jsonl")):
            s = score_file(f)
            if s is None:
                continue
            any_found = True
            print(f"{prot.name}\t{f.relative_to(bench)}\t{s['n_scenarios']}\t{s['shipped_pct']}\t{s['false_ship_pct']}\t{s['reject_pct']}\t{s['mean_tokens_per_bundle']}\t{s['tokens_per_correct_shipped']}")

    if not any_found:
        print("No run files found under runs/<protocol>/*.jsonl", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()

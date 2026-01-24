# Contributing

Goal: build an interoperable “forest” of TruthCert variants.

## Add a validator
Edit `/validators/validator_registry.yaml` and include:
- what it catches, expected IO, expected cost
- **false_positive_rate < 0.05**
- **coverage > 0.50**
- **no regression on existing checks**
- human approval + docs

## Add a pack
Create `/packs/<PACK_ID>/` with `pack.yaml`, `validators.yaml`, `corruptions.yaml`.

## Add a benchmark suite
Create a folder under `/benchmarks/` with tasks + gold + scoring.

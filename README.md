# Real-RCT Benchmark v0.1 (skeleton)

Fill these files:
- `papers.csv` : paper_id,title,year,oa_url,pdf_sha256
- `tasks.jsonl` : one strict extraction task per paper (scope-locked)
- `gold.csv` : gold answers + page/table references

Place model outputs into:
- `runs/<protocol>/*.jsonl`

Score with:
```bash
python ../../tools/score_contract_v1.py .
```

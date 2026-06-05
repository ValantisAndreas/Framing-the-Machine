# RQ3 branch notes

This branch is based on `main` and adds **RQ3: Syntax Feature Relevance** without changing `analyze_data.py`.

This version is **NYT-only** and uses the same Kaggle prerequisite as the main branch:

- Download `nyt-metadata.csv` from Kaggle: NYT Articles 2.1M+ (2000-Present).
- Place `nyt-metadata.csv` directly in the repository root.

Added file:

- `rq3_syntax_features.py`: NYT-only RQ3 analysis module.

Updated files:

- `README.md`: NYT-only RQ3 instructions.
- `requirements.txt`: adds VADER/scipy dependencies needed for RQ3.

Design compatibility:

- Imports shared constants and cleaning logic from `analyze_data.py`.
- Uses the same year window, AI keyword regex, and exclusion logic as RQ1.
- Preserves `clean_headline` and `abstract` separately for RQ3 syntax analysis.
- Writes all outputs under `results/`, matching the existing project style.

Run:

```bash
python rq3_syntax_features.py
```

Main output:

```text
results/rq3_nyt_syntax_feature_relevance.csv
```

# Framing the Machine

*"Framing the Machine: A 25-Year Temporal and Sentiment Analysis of Artificial Intelligence Representation in Mainstream Journalism Across the US and EU (2000-2025)"*.

This branch adds **RQ3: Syntax Feature Relevance** for the NYT-only Kaggle dataset workflow.

## Prerequisites

1. **Python** installed on your system.
2. **Kaggle Dataset:** Download the `nyt-metadata.csv` file from Kaggle: **NYT Articles 2.1M+ (2000-Present)**. Place the extracted `nyt-metadata.csv` file directly into the root directory of this project.

## Install dependencies

```bash
pip install -r requirements.txt
```

## RQ1: Thematic Evolution

```bash
python analyze_data.py
```

## RQ3: Syntax Feature Relevance

RQ3 asks which linguistic indicators in NYT headlines and abstracts are most associated with emotional intensity.

The RQ3 script uses the same AI keyword filter, year window, and exclusion logic as the main/RQ1 pipeline, but it preserves headlines and abstracts separately so syntax features can be measured.

Run RQ3 with the default Kaggle file in the project root:

```bash
python rq3_syntax_features.py
```

Or explicitly pass the file path:

```bash
python rq3_syntax_features.py --input nyt-metadata.csv
```

Optional: change the minimum correlation threshold for selecting features:

```bash
python rq3_syntax_features.py --min-abs-corr 0.08
```

## RQ3 outputs

The script writes these files to `results/`:

- `rq3_nyt_dataset_with_syntax_features.csv` — article-level NYT dataset with sentiment scores and syntax features.
- `rq3_nyt_syntax_feature_relevance.csv` — main RQ3 correlation-based feature selection table.
- `rq3_nyt_syntax_feature_results.json` — structured RQ3 summary for reporting.
- `rq3_nyt_top_feature_relevance.png` — top linguistic indicators by absolute Spearman correlation.
- `rq3_nyt_yearly_emotional_intensity.png` — yearly emotional-intensity trend.

The core columns in `rq3_nyt_syntax_feature_relevance.csv` are:

- `feature`
- `spearman_corr_with_emotional_intensity`
- `p_value_intensity`
- `fdr_q_value_intensity`
- `absolute_relevance`
- `selected_feature`

A feature is marked as selected when it satisfies both:

```text
absolute_relevance >= 0.10
fdr_q_value_intensity < 0.05
```

## Git workflow

From `main`, create the RQ3 branch:

```bash
git checkout main
git checkout -b Rq3
```

Then add the RQ3 files and commit:

```bash
git add rq3_syntax_features.py README.md requirements.txt RQ3_BRANCH_NOTES.md
git commit -m "Add NYT-only RQ3 syntax feature relevance analysis"
```

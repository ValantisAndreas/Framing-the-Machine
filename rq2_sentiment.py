import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reuse RQ1 primitives without altering them.
from analyze_data import (
    DATA_FILE,
    KEYWORD_REGEX,
    clean_headline,
    standardize_phrases,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Epoch boundaries. Aligned with the RQ1 (analyze_data.py) three-era split so
# the two analyses are directly comparable: pre-deep-learning, deep-learning
# expansion, and the generative-AI era.
EPOCHS = [
    ("Era1 (2000-2011)", 2000, 2011),   # Pre-deep-learning
    ("Era2 (2012-2018)", 2012, 2018),   # Deep-learning expansion (post-AlexNet)
    ("Era3 (2019-2025)", 2019, 2025),   # Generative-AI integration era
]

# Pairwise epoch comparisons of interest for drift testing.
EPOCH_PAIRS = [
    ("Era1 (2000-2011)", "Era2 (2012-2018)"),
    ("Era2 (2012-2018)", "Era3 (2019-2025)"),  # core post-GenAI comparison
    ("Era1 (2000-2011)", "Era3 (2019-2025)"),  # full-range drift
]

# Compound-score thresholds (VADER convention).
STRONG_POS = 0.5
STRONG_NEG = -0.5
NEUTRAL_BAND = 0.05

RESULTS_DIR = "results"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_and_filter_data_rq2(filepath):
    """Load the NYT dataset and keep the abstract and headline columns
    separately, applying the same year window, exclusion filters and AI
    keyword filter used in RQ1.

    Returns a DataFrame with columns: year, clean_headline, abstract.
    """
    print("[*] Preprocessing data for RQ2 (sentiment)...")
    filtered_chunks = []
    chunk_iterator = pd.read_csv(filepath, chunksize=50000, low_memory=False)

    # Mirror the RQ1 exclusion of non-news formats and off-topic sections.
    exclude_materials = {'Schedule', 'Paid Death Notice', 'Obituary', 'Letter',
                         'Review', 'List', 'Summary', 'Brief'}
    exclude_sections = {'Sports', 'Sports Desk', 'Crosswords', 'Arts', 'Theater',
                        'Movies', 'Travel', 'Real Estate'}

    for chunk in chunk_iterator:
        chunk.columns = [col.lower().strip() for col in chunk.columns]
        if 'pub_date' not in chunk.columns or 'abstract' not in chunk.columns:
            continue

        chunk['year'] = chunk['pub_date'].astype(str).str[:4]
        chunk = chunk[chunk['year'].str.isnumeric()]
        chunk = chunk.astype({'year': int})
        chunk = chunk[(chunk['year'] >= 2000) & (chunk['year'] <= 2025)]
        if chunk.empty:
            continue

        if 'type_of_material' in chunk.columns:
            chunk = chunk[~chunk['type_of_material'].isin(exclude_materials)]
        if 'section_name' in chunk.columns:
            chunk = chunk[~chunk['section_name'].astype(str).str.strip()
                          .str.title().isin(exclude_sections)]
        if chunk.empty:
            continue

        headline_col = 'headline' if 'headline' in chunk.columns else chunk.columns[0]
        chunk['clean_headline'] = chunk[headline_col].apply(clean_headline)
        chunk['abstract'] = chunk['abstract'].fillna('')

        # Filter using the same standardized-text keyword match as RQ1, so the
        # RQ2 corpus is identical to the RQ1 corpus.
        match_text = (chunk['clean_headline'] + " " + chunk['abstract']).apply(standardize_phrases)
        is_ai_related = match_text.str.contains(KEYWORD_REGEX, na=False)
        matched = chunk[is_ai_related].copy()

        if not matched.empty:
            filtered_chunks.append(matched[['year', 'clean_headline', 'abstract']])

    if not filtered_chunks:
        raise ValueError("No matching records found.")

    final_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"[+] Extracted {len(final_df)} filtered articles.")
    return final_df


# --------------------------------------------------------------------------- #
# Sentiment scoring
# --------------------------------------------------------------------------- #

def compute_sentiment(df):
    """Add VADER compound scores for abstract (primary) and headline (secondary)."""
    print("[*] Scoring sentiment with VADER...")
    analyzer = SentimentIntensityAnalyzer()

    def score(text):
        if not isinstance(text, str) or not text.strip():
            return np.nan
        return analyzer.polarity_scores(text)['compound']

    df['abstract_sentiment'] = df['abstract'].apply(score)
    df['headline_sentiment'] = df['clean_headline'].apply(score)
    return df


def assign_epochs(df):
    """Label each article with its epoch based on publication year."""
    df['epoch'] = 'Unknown'
    for label, lo, hi in EPOCHS:
        df.loc[(df['year'] >= lo) & (df['year'] <= hi), 'epoch'] = label
    return df


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def epoch_descriptive_stats(scores):
    """Descriptive + polarization statistics for one epoch's compound scores."""
    s = scores.dropna()
    n = len(s)
    if n == 0:
        return None
    return {
        'n': int(n),
        'mean': float(s.mean()),
        'std': float(s.std(ddof=1)) if n > 1 else 0.0,
        'median': float(s.median()),
        'iqr': float(s.quantile(0.75) - s.quantile(0.25)),
        'pct_strong_positive': float((s > STRONG_POS).mean() * 100),
        'pct_strong_negative': float((s < STRONG_NEG).mean() * 100),
        'pct_neutral': float((s.abs() < NEUTRAL_BAND).mean() * 100),
    }


def rank_biserial_effect_size(u_stat, n1, n2):
    """Rank-biserial correlation as an effect size for Mann-Whitney U."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def run_significance_tests(df, score_col):
    """Mann-Whitney U tests between epoch pairs for the given score column."""
    results = []
    for a, b in EPOCH_PAIRS:
        sa = df.loc[df['epoch'] == a, score_col].dropna()
        sb = df.loc[df['epoch'] == b, score_col].dropna()
        if len(sa) == 0 or len(sb) == 0:
            continue
        u_stat, p_val = mannwhitneyu(sa, sb, alternative='two-sided')
        results.append({
            'epoch_a': a,
            'epoch_b': b,
            'n_a': int(len(sa)),
            'n_b': int(len(sb)),
            'mean_a': float(sa.mean()),
            'mean_b': float(sb.mean()),
            'std_a': float(sa.std(ddof=1)) if len(sa) > 1 else 0.0,
            'std_b': float(sb.std(ddof=1)) if len(sb) > 1 else 0.0,
            'u_statistic': float(u_stat),
            'p_value': float(p_val),
            'significant_p05': bool(p_val < 0.05),
            'rank_biserial_effect': float(rank_biserial_effect_size(u_stat, len(sa), len(sb))),
        })
    return results


def yearly_aggregates(df, score_col):
    """Per-year mean and std of sentiment (polarization signal over time)."""
    grp = df.dropna(subset=[score_col]).groupby('year')[score_col]
    out = pd.DataFrame({
        'year': grp.mean().index.astype(int),
        'mean': grp.mean().values,
        'std': grp.std(ddof=1).values,
        'n': grp.size().values,
    })
    return out.sort_values('year').reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

# Event markers used as vertical reference lines on time-series plots.
EVENT_LINES = [
    (2012, "AlexNet (2012)"),
    (2022, "ChatGPT (2022)"),
]

EPOCH_ORDER = [label for label, _, _ in EPOCHS]


def plot_yearly_mean(yearly, path):
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=yearly, x='year', y='mean', marker='o', color='steelblue')
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    for yr, lbl in EVENT_LINES:
        plt.axvline(yr, color='firebrick', linestyle='--', linewidth=1)
        plt.text(yr + 0.1, plt.ylim()[1] * 0.9, lbl, color='firebrick', fontsize=9)
    plt.title("Yearly Mean Abstract Sentiment of NYT AI Coverage (2000-2025) (RQ2)",
              fontsize=14, weight='bold')
    plt.ylabel("Mean VADER Compound Score")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"    [+] Saved: '{path}'")


def plot_epoch_distributions(df, score_col, path):
    plt.figure(figsize=(11, 6))
    order = [e for e in EPOCH_ORDER if e in df['epoch'].unique()]
    sns.violinplot(data=df.dropna(subset=[score_col]), x='epoch', y=score_col,
                   order=order, inner='box', palette='muted', cut=0)
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    plt.title("Sentiment Distribution per Tech Epoch (Spread = Polarization) (RQ2)",
              fontsize=14, weight='bold')
    plt.ylabel("VADER Compound Score")
    plt.xlabel("Tech Epoch")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"    [+] Saved: '{path}'")


def plot_sentiment_proportions(df, score_col, path):
    order = [e for e in EPOCH_ORDER if e in df['epoch'].unique()]
    rows = []
    for ep in order:
        s = df.loc[df['epoch'] == ep, score_col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            'epoch': ep,
            'Strongly Positive': (s > STRONG_POS).mean() * 100,
            'Neutral / Mild': ((s >= STRONG_NEG) & (s <= STRONG_POS)).mean() * 100,
            'Strongly Negative': (s < STRONG_NEG).mean() * 100,
        })
    prop = pd.DataFrame(rows).set_index('epoch')

    prop.plot(kind='bar', stacked=True, figsize=(11, 6),
              color=['#2a9d8f', '#cccccc', '#e76f51'])
    plt.title("Polarization: Sentiment Category Proportions per Epoch (RQ2)",
              fontsize=14, weight='bold')
    plt.ylabel("Percentage of Articles")
    plt.xlabel("Tech Epoch")
    plt.xticks(rotation=0)
    plt.legend(title="Sentiment Category", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"    [+] Saved: '{path}'")


def plot_yearly_polarization(yearly, path):
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=yearly, x='year', y='std', marker='o', color='darkorange')
    for yr, lbl in EVENT_LINES:
        plt.axvline(yr, color='firebrick', linestyle='--', linewidth=1)
        plt.text(yr + 0.1, plt.ylim()[1] * 0.92, lbl, color='firebrick', fontsize=9)
    plt.title("Yearly Sentiment Dispersion (Std Dev) = Polarization Trend (RQ2)",
              fontsize=14, weight='bold')
    plt.ylabel("Std Dev of VADER Compound Score")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"    [+] Saved: '{path}'")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if not os.path.exists(DATA_FILE):
        print(f"[!] Error: '{DATA_FILE}' not found in the root directory.")
        print("    Download it from the Kaggle dataset listed in README.md.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time = time.time()

    df = load_and_filter_data_rq2(DATA_FILE)
    df = compute_sentiment(df)
    df = assign_epochs(df)
    df = df[df['epoch'] != 'Unknown']

    order = [e for e in EPOCH_ORDER if e in df['epoch'].unique()]

    # --- Descriptive statistics per epoch (primary = abstract) --------------- #
    print("\n" + "=" * 80)
    print("RQ2: SENTIMENT DRIFT & POLARIZATION (primary signal = abstract)")
    print("=" * 80)

    epoch_stats = {'abstract': {}, 'headline': {}}
    for ep in order:
        sub = df[df['epoch'] == ep]
        epoch_stats['abstract'][ep] = epoch_descriptive_stats(sub['abstract_sentiment'])
        epoch_stats['headline'][ep] = epoch_descriptive_stats(sub['headline_sentiment'])

    summary = pd.DataFrame({ep: epoch_stats['abstract'][ep] for ep in order}).T
    print("\n[Abstract sentiment per epoch]")
    print(summary.to_string())

    # --- Significance tests -------------------------------------------------- #
    print("\n[Mann-Whitney U tests on abstract sentiment]")
    tests_abstract = run_significance_tests(df, 'abstract_sentiment')
    tests_headline = run_significance_tests(df, 'headline_sentiment')
    for t in tests_abstract:
        flag = "SIGNIFICANT" if t['significant_p05'] else "n.s."
        print(f"  {t['epoch_a']} vs {t['epoch_b']}: "
              f"U={t['u_statistic']:.0f}, p={t['p_value']:.3e} [{flag}], "
              f"mean {t['mean_a']:.3f}->{t['mean_b']:.3f}, "
              f"std {t['std_a']:.3f}->{t['std_b']:.3f}, "
              f"effect r={t['rank_biserial_effect']:.3f}")

    # --- Yearly aggregates --------------------------------------------------- #
    yearly_abstract = yearly_aggregates(df, 'abstract_sentiment')

    elapsed = time.time() - start_time
    print(f"\n[+] Processing completed in {elapsed:.2f} seconds.")

    # --- Persist JSON -------------------------------------------------------- #
    results_json = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time_seconds': float(f"{elapsed:.2f}"),
        'total_documents': int(len(df)),
        'primary_signal': 'abstract',
        'epoch_definitions': {label: [lo, hi] for label, lo, hi in EPOCHS},
        'epoch_document_counts': {ep: int((df['epoch'] == ep).sum()) for ep in order},
        'descriptive_statistics': epoch_stats,
        'significance_tests': {
            'abstract': tests_abstract,
            'headline': tests_headline,
        },
        'yearly_abstract_sentiment': yearly_abstract.to_dict(orient='records'),
    }
    json_path = os.path.join(RESULTS_DIR, "rq2_sentiment_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"    [+] Saved quantitative results to: '{json_path}'")

    # --- Visualizations ------------------------------------------------------ #
    print("[*] Generating RQ2 plots...")
    sns.set_theme(style="whitegrid")
    plot_yearly_mean(yearly_abstract, os.path.join(RESULTS_DIR, "rq2_yearly_mean_sentiment.png"))
    plot_epoch_distributions(df, 'abstract_sentiment', os.path.join(RESULTS_DIR, "rq2_epoch_distributions.png"))
    plot_sentiment_proportions(df, 'abstract_sentiment', os.path.join(RESULTS_DIR, "rq2_sentiment_proportions.png"))
    plot_yearly_polarization(yearly_abstract, os.path.join(RESULTS_DIR, "rq2_yearly_polarization.png"))
    print("[+] RQ2 analysis complete.\n")


if __name__ == "__main__":
    main()

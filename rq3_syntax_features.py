import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# RQ3 depends on the main branch pipeline instead of duplicating its constants.
# This keeps the NYT article filtering aligned with RQ1/RQ2.
from analyze_data import (
    DATA_FILE,
    KEYWORD_REGEX,
    clean_headline,
    standardize_phrases,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RESULTS_DIR = "results"
YEAR_MIN = 2000
YEAR_MAX = 2025
CHUNK_SIZE = 50000

# Correlation-based feature-selection settings.
MIN_ABS_CORR = 0.10
ALPHA = 0.05

# VADER compound-score thresholds used only for descriptive output.
STRONG_POS = 0.5
STRONG_NEG = -0.5

# Epoch boundaries aligned with analyze_data.py / RQ1 and rq2_sentiment.py.
EPOCHS = [
    ("Era1 (2000-2011)", 2000, 2011),
    ("Era2 (2012-2018)", 2012, 2018),
    ("Era3 (2019-2025)", 2019, 2025),
]

# Exclusions copied from the main branch so RQ3 uses the same NYT corpus logic.
EXCLUDE_MATERIALS = {
    "Schedule", "Paid Death Notice", "Obituary", "Letter", "Review",
    "List", "Summary", "Brief",
}
EXCLUDE_SECTIONS = {
    "Sports", "Sports Desk", "Crosswords", "Arts", "Theater", "Movies",
    "Travel", "Real Estate",
}

# --------------------------------------------------------------------------- #
# Linguistic dictionaries for interpretable syntax / rhetoric features
# --------------------------------------------------------------------------- #

SUPERLATIVE_TERMS = [
    "best", "worst", "biggest", "smallest", "largest", "greatest",
    "most", "least", "first", "last", "only", "ultimate", "unprecedented",
    "historic", "revolutionary", "record", "record-breaking",
]

SPECULATIVE_TERMS = [
    "may", "might", "could", "would", "should", "will", "can",
    "possible", "possibly", "potential", "potentially", "likely", "unlikely",
    "future", "predict", "prediction", "forecast", "expect", "expected",
]

RISK_TERMS = [
    "risk", "risks", "threat", "threats", "danger", "dangerous", "fear",
    "fears", "warning", "warn", "warns", "bias", "biased", "surveillance",
    "misinformation", "disinformation", "replace", "replaces", "job loss",
    "layoff", "layoffs", "existential", "doomsday", "harm", "harms",
]

PROGRESS_TERMS = [
    "breakthrough", "innovation", "innovative", "advance", "advances",
    "progress", "improve", "improves", "benefit", "benefits", "boost",
    "opportunity", "growth", "transform", "transforms", "revolutionize",
    "revolutionizes", "productivity", "efficient", "efficiency",
]

CLICKBAIT_TERMS = [
    "why", "how", "what", "when", "reveals", "revealed", "shocking",
    "surprising", "secret", "secrets", "experts say", "you need to know",
    "the truth", "what happens", "here's", "here is", "this is why",
]

MODAL_PATTERN = re.compile(
    r"\b(?:may|might|could|would|should|will|can|must|shall)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\b\w+\b")
SENTENCE_PATTERN = re.compile(r"[.!?]+")

FEATURE_COLUMNS = [
    "headline_word_count",
    "abstract_word_count",
    "combined_word_count",
    "headline_char_count",
    "abstract_char_count",
    "headline_avg_word_length",
    "abstract_avg_word_length",
    "abstract_sentence_count",
    "headline_has_question_mark",
    "abstract_has_question_mark",
    "headline_has_exclamation_mark",
    "abstract_has_exclamation_mark",
    "headline_has_colon",
    "headline_has_number",
    "abstract_has_number",
    "headline_comma_count",
    "abstract_comma_count",
    "superlative_count",
    "speculative_count",
    "modal_verb_count",
    "risk_term_count",
    "progress_term_count",
    "clickbait_term_count",
]

# --------------------------------------------------------------------------- #
# Data loading: NYT Kaggle metadata only
# --------------------------------------------------------------------------- #

def load_and_filter_nyt_for_rq3(filepath):
    """Load Kaggle nyt-metadata.csv and keep headline/abstract information.

    This mirrors the filtering logic in analyze_data.py, but preserves the
    headline and abstract separately because RQ3 measures syntax features on
    both fields.

    Expected input: nyt-metadata.csv in the project root.
    Returns columns: year, date, clean_headline, abstract, combined_text.
    """
    print(f"[*] Preprocessing NYT data for RQ3: '{filepath}'")
    filtered_chunks = []

    chunk_iterator = pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False)

    for i, chunk in enumerate(chunk_iterator, start=1):
        chunk.columns = [col.lower().strip() for col in chunk.columns]
        if "pub_date" not in chunk.columns or "abstract" not in chunk.columns:
            continue

        chunk["year"] = chunk["pub_date"].astype(str).str[:4]
        chunk = chunk[chunk["year"].str.isnumeric()]
        chunk = chunk.astype({"year": int})
        chunk = chunk[(chunk["year"] >= YEAR_MIN) & (chunk["year"] <= YEAR_MAX)]
        if chunk.empty:
            continue

        if "type_of_material" in chunk.columns:
            chunk = chunk[~chunk["type_of_material"].isin(EXCLUDE_MATERIALS)]
        if "section_name" in chunk.columns:
            chunk = chunk[
                ~chunk["section_name"].astype(str).str.strip().str.title().isin(EXCLUDE_SECTIONS)
            ]
        if chunk.empty:
            continue

        headline_col = "headline" if "headline" in chunk.columns else chunk.columns[0]
        chunk["clean_headline"] = chunk[headline_col].apply(clean_headline)
        chunk["abstract"] = chunk["abstract"].fillna("").astype(str)
        chunk["date"] = pd.to_datetime(chunk["pub_date"], errors="coerce")

        # Same AI keyword filtering as main/RQ1.
        match_text = (chunk["clean_headline"] + " " + chunk["abstract"]).apply(standardize_phrases)
        matched = chunk[match_text.str.contains(KEYWORD_REGEX, na=False)].copy()

        if matched.empty:
            continue

        matched["combined_text"] = (
            matched["clean_headline"].fillna("") + ". " + matched["abstract"].fillna("")
        )

        keep = ["year", "date", "clean_headline", "abstract", "combined_text"]
        optional = [c for c in ["web_url", "section_name", "word_count", "type_of_material"] if c in matched.columns]
        filtered_chunks.append(matched[keep + optional])

    if not filtered_chunks:
        raise ValueError("No matching AI-related NYT records found for RQ3.")

    df = pd.concat(filtered_chunks, ignore_index=True)
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date", "clean_headline"])
    df["outlet"] = "NYT"
    print(f"[+] Extracted {len(df)} NYT articles after RQ1-compatible filtering.")
    return df.reset_index(drop=True)


def assign_epochs(df):
    df["epoch"] = "Unknown"
    for label, lo, hi in EPOCHS:
        df.loc[(df["year"] >= lo) & (df["year"] <= hi), "epoch"] = label
    return df[df["epoch"] != "Unknown"].copy()

# --------------------------------------------------------------------------- #
# Sentiment and feature extraction
# --------------------------------------------------------------------------- #

def compute_sentiment(df):
    """Add VADER sentiment and emotional-intensity scores.

    Emotional intensity is abs(compound sentiment), so highly positive and
    highly negative articles are both treated as emotionally intense.
    """
    print("[*] Scoring headline/abstract sentiment with VADER...")
    analyzer = SentimentIntensityAnalyzer()

    def score(text):
        if not isinstance(text, str) or not text.strip():
            return np.nan
        return analyzer.polarity_scores(text)["compound"]

    df["headline_sentiment"] = df["clean_headline"].apply(score)
    df["abstract_sentiment"] = df["abstract"].apply(score)
    df["combined_sentiment"] = df["combined_text"].apply(score)

    df["headline_emotional_intensity"] = df["headline_sentiment"].abs()
    df["abstract_emotional_intensity"] = df["abstract_sentiment"].abs()
    df["combined_emotional_intensity"] = df["combined_sentiment"].abs()

    df["positive_extreme"] = (df["combined_sentiment"] >= STRONG_POS).astype(int)
    df["negative_extreme"] = (df["combined_sentiment"] <= STRONG_NEG).astype(int)
    df["any_extreme"] = ((df["positive_extreme"] == 1) | (df["negative_extreme"] == 1)).astype(int)
    return df


def _words(text):
    if not isinstance(text, str):
        return []
    return WORD_PATTERN.findall(text.lower())


def _sentence_count(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    n = len([s for s in SENTENCE_PATTERN.split(text) if s.strip()])
    return max(n, 1)


def _count_terms(text, terms):
    if not isinstance(text, str):
        return 0
    text_l = text.lower()
    return sum(1 for term in terms if term in text_l)


def _avg_word_length(words):
    return float(np.mean([len(w) for w in words])) if words else 0.0


def extract_linguistic_features(row):
    headline = row.get("clean_headline", "") or ""
    abstract = row.get("abstract", "") or ""
    combined = f"{headline} {abstract}"

    hw = _words(headline)
    aw = _words(abstract)
    cw = _words(combined)

    return pd.Series({
        # Length and structure
        "headline_word_count": len(hw),
        "abstract_word_count": len(aw),
        "combined_word_count": len(cw),
        "headline_char_count": len(headline),
        "abstract_char_count": len(abstract),
        "headline_avg_word_length": _avg_word_length(hw),
        "abstract_avg_word_length": _avg_word_length(aw),
        "abstract_sentence_count": _sentence_count(abstract),

        # Punctuation and formatting
        "headline_has_question_mark": int("?" in headline),
        "abstract_has_question_mark": int("?" in abstract),
        "headline_has_exclamation_mark": int("!" in headline),
        "abstract_has_exclamation_mark": int("!" in abstract),
        "headline_has_colon": int(":" in headline),
        "headline_has_number": int(bool(re.search(r"\d", headline))),
        "abstract_has_number": int(bool(re.search(r"\d", abstract))),
        "headline_comma_count": headline.count(","),
        "abstract_comma_count": abstract.count(","),

        # Interpretable rhetorical indicators
        "superlative_count": _count_terms(combined, SUPERLATIVE_TERMS),
        "speculative_count": _count_terms(combined, SPECULATIVE_TERMS),
        "modal_verb_count": len(MODAL_PATTERN.findall(combined)),
        "risk_term_count": _count_terms(combined, RISK_TERMS),
        "progress_term_count": _count_terms(combined, PROGRESS_TERMS),
        "clickbait_term_count": _count_terms(combined, CLICKBAIT_TERMS),
    })


def add_linguistic_features(df):
    print("[*] Extracting RQ3 syntax and rhetoric features...")
    features = df.apply(extract_linguistic_features, axis=1)
    return pd.concat([df, features], axis=1)

# --------------------------------------------------------------------------- #
# Correlation-based feature selection
# --------------------------------------------------------------------------- #

def safe_spearman(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan
    corr, p_val = spearmanr(tmp["x"], tmp["y"])
    return float(corr), float(p_val)


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction. NaNs are preserved."""
    p = np.array(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return q.tolist()

    idx_valid = np.where(valid)[0]
    order = idx_valid[np.argsort(p[valid])]
    ranked_p = p[order]
    m = len(ranked_p)
    adjusted = np.empty(m, dtype=float)

    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked_p[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = min(prev, 1.0)

    q[order] = adjusted
    return q.tolist()


def correlation_feature_relevance(df, target_col="combined_emotional_intensity", min_abs_corr=MIN_ABS_CORR):
    print(f"[*] Running Spearman feature relevance against '{target_col}'...")
    rows = []

    for feature in FEATURE_COLUMNS:
        corr_intensity, p_intensity = safe_spearman(df[feature], df[target_col])
        corr_direction, p_direction = safe_spearman(df[feature], df["combined_sentiment"])
        corr_headline_int, p_headline_int = safe_spearman(df[feature], df["headline_emotional_intensity"])
        corr_abstract_int, p_abstract_int = safe_spearman(df[feature], df["abstract_emotional_intensity"])

        rows.append({
            "feature": feature,
            "target": target_col,
            "spearman_corr_with_emotional_intensity": corr_intensity,
            "p_value_intensity": p_intensity,
            "spearman_corr_with_sentiment_direction": corr_direction,
            "p_value_sentiment_direction": p_direction,
            "spearman_corr_with_headline_intensity": corr_headline_int,
            "p_value_headline_intensity": p_headline_int,
            "spearman_corr_with_abstract_intensity": corr_abstract_int,
            "p_value_abstract_intensity": p_abstract_int,
            "absolute_relevance": abs(corr_intensity) if not np.isnan(corr_intensity) else np.nan,
        })

    out = pd.DataFrame(rows)
    out["fdr_q_value_intensity"] = benjamini_hochberg(out["p_value_intensity"].tolist())
    out["selected_feature"] = (
        (out["absolute_relevance"] >= min_abs_corr) &
        (out["fdr_q_value_intensity"] < ALPHA)
    )
    out = out.sort_values("absolute_relevance", ascending=False, na_position="last")
    return out.reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #

def plot_top_feature_relevance(feature_relevance, path, top_n=12):
    plot_df = feature_relevance.dropna(subset=["absolute_relevance"]).head(top_n).copy()
    if plot_df.empty:
        print("    [!] Skipping top-feature plot: no valid correlations.")
        return

    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=plot_df,
        y="feature",
        x="absolute_relevance",
        hue="selected_feature",
        dodge=False,
    )
    plt.title("Top Syntax Features Associated with NYT Emotional Intensity (RQ3)",
              fontsize=14, weight="bold")
    plt.xlabel("Absolute Spearman Correlation with Emotional Intensity")
    plt.ylabel("Feature")
    plt.legend(title="Selected by FDR + effect cutoff", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"    [+] Saved: '{path}'")


def plot_yearly_emotional_intensity(df, path):
    yearly = (
        df.dropna(subset=["combined_emotional_intensity"])
          .groupby("year")["combined_emotional_intensity"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    if yearly.empty:
        print("    [!] Skipping yearly intensity plot: no valid sentiment scores.")
        return

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=yearly, x="year", y="mean", marker="o")
    for yr, label in [(2012, "AlexNet (2012)"), (2022, "ChatGPT (2022)")]:
        plt.axvline(yr, linestyle="--", linewidth=1)
        y_max = yearly["mean"].max() if yearly["mean"].notna().any() else 1
        plt.text(yr + 0.1, y_max * 0.95, label, fontsize=9)
    plt.title("Yearly Mean Emotional Intensity in NYT AI Headlines/Abstracts (RQ3)",
              fontsize=14, weight="bold")
    plt.ylabel("Mean |VADER Compound Score|")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"    [+] Saved: '{path}'")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3: NYT syntax feature relevance for emotional intensity in headlines/abstracts."
    )
    parser.add_argument(
        "--input",
        default=DATA_FILE,
        help="Path to Kaggle NYT metadata CSV. Default: nyt-metadata.csv in the project root.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory where RQ3 CSV/JSON/PNG outputs are saved.",
    )
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=MIN_ABS_CORR,
        help="Minimum absolute Spearman correlation for selecting a feature.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: '{args.input}' not found in the project root.")
        print("    Download nyt-metadata.csv from Kaggle and place it in the root directory.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    start = time.time()

    df = load_and_filter_nyt_for_rq3(args.input)
    df = assign_epochs(df)
    df = compute_sentiment(df)
    df = add_linguistic_features(df)

    feature_relevance = correlation_feature_relevance(
        df,
        target_col="combined_emotional_intensity",
        min_abs_corr=args.min_abs_corr,
    )

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("RQ3: NYT SYNTAX FEATURE RELEVANCE")
    print("=" * 80)
    print(f"Documents: {len(df)}")
    print("Corpus: NYT Kaggle metadata only")
    print("\n[Top feature relevance]")
    cols_to_print = [
        "feature", "spearman_corr_with_emotional_intensity", "p_value_intensity",
        "fdr_q_value_intensity", "absolute_relevance", "selected_feature",
    ]
    print(feature_relevance[cols_to_print].head(15).to_string(index=False))

    selected = feature_relevance[feature_relevance["selected_feature"]]
    print(f"\n[+] Selected {len(selected)} features using |rho| >= {args.min_abs_corr} and FDR q < {ALPHA}.")
    print(f"[+] Processing completed in {elapsed:.2f} seconds.")

    dataset_path = os.path.join(args.results_dir, "rq3_nyt_dataset_with_syntax_features.csv")
    relevance_path = os.path.join(args.results_dir, "rq3_nyt_syntax_feature_relevance.csv")
    json_path = os.path.join(args.results_dir, "rq3_nyt_syntax_feature_results.json")

    df.to_csv(dataset_path, index=False)
    feature_relevance.to_csv(relevance_path, index=False)

    results_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": float(f"{elapsed:.2f}"),
        "corpus": "NYT Kaggle metadata only",
        "input_file": args.input,
        "total_documents": int(len(df)),
        "epoch_document_counts": {str(k): int(v) for k, v in df["epoch"].value_counts().to_dict().items()},
        "primary_target": "combined_emotional_intensity",
        "feature_selection_rule": {
            "method": "Spearman correlation + Benjamini-Hochberg FDR",
            "min_abs_corr": args.min_abs_corr,
            "alpha_fdr": ALPHA,
        },
        "feature_relevance": feature_relevance.to_dict(orient="records"),
        "selected_features": selected["feature"].tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)

    print(f"    [+] Saved dataset to: '{dataset_path}'")
    print(f"    [+] Saved feature relevance table to: '{relevance_path}'")
    print(f"    [+] Saved JSON results to: '{json_path}'")

    print("[*] Generating RQ3 plots...")
    sns.set_theme(style="whitegrid")
    plot_top_feature_relevance(
        feature_relevance,
        os.path.join(args.results_dir, "rq3_nyt_top_feature_relevance.png"),
    )
    plot_yearly_emotional_intensity(
        df,
        os.path.join(args.results_dir, "rq3_nyt_yearly_emotional_intensity.png"),
    )
    print("[+] RQ3 NYT-only analysis complete.\n")


if __name__ == "__main__":
    main()

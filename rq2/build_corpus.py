"""
rq2/build_corpus.py
Framing the Machine — RQ2 corpus builder (NYT only).

Purpose
-------
Produce a filtered, sentiment-ready corpus of AI-related NYT articles that is
*identical in membership* to the RQ1 corpus, but enriched with the information
RQ2 needs and RQ1 discards:

  1. `abstract` and `clean_headline` kept as SEPARATE columns
     (RQ1's loader collapses them into one `combined_text`; sentiment needs them apart).
  2. A PRECISE publication timestamp converted to the New York timezone
     (`pub_datetime_ny`) — RQ1 keeps only the 4-digit year. This lets us split the
     generative-AI epoch at the exact ChatGPT public-release date (2022-11-30, NY time).

Self-contained by design
------------------------
The text-filtering primitives (KEYWORDS / KEYWORD_REGEX / standardize_phrases /
clean_headline) are MIRRORED verbatim from the teammate's `analyze_data.py`
(RQ1) so the RQ2 corpus membership matches RQ1 exactly. They are copied rather
than imported on purpose: importing `analyze_data` pulls in gensim, which has no
prebuilt wheel for Python 3.14 and fails to compile. RQ2 needs only pandas.

  >>> IF RQ1's keyword set / phrase list changes, re-sync the marked block below. <<<

Output: rq2/rq2_corpus.csv
"""

import os
import re
import ast

import pandas as pd

# The raw Kaggle dump lives in the project root (same file RQ1 uses).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "nyt-metadata.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rq2_corpus.csv")

NY_TZ = "America/New_York"

# ChatGPT public release: 30 Nov 2022. We anchor the within-Era3 split to the
# start of that day in New York local time (NYT's home timezone). DST is handled
# automatically by the zoneinfo database via pandas.
CHATGPT_RELEASE_NY = pd.Timestamp("2022-11-30 00:00:00", tz=NY_TZ)

# Epoch boundaries — identical to RQ1 (analyze_data.py) and the project proposal.
EPOCHS = [
    ("Era1", 2000, 2011),   # Pre-deep-learning
    ("Era2", 2012, 2018),   # Deep-learning expansion (post-AlexNet)
    ("Era3", 2019, 2025),   # Generative-AI integration
]

EXCLUDE_MATERIALS = {
    'Schedule', 'Paid Death Notice', 'Obituary', 'Letter',
    'Review', 'List', 'Summary', 'Brief',
}
EXCLUDE_SECTIONS = {
    'Sports', 'Sports Desk', 'Crosswords', 'Arts',
    'Theater', 'Movies', 'Travel', 'Real Estate',
}

CHUNK_SIZE = 50_000

# ═════════════════════════════════════════════════════════════════════════════
# MIRRORED FROM analyze_data.py (RQ1) — keep in sync if RQ1's filter changes.
# ═════════════════════════════════════════════════════════════════════════════

KEYWORDS = [
    # Core AI & ML
    r'artificial\s+intelligence', r'machine\s+learning', r'deep\s+learning',
    r'neural\s+network',

    # Robotics & Labor
    r'automation', r'robot',

    # Generative AI & Platforms
    r'chatgpt', r'generative\s+ai', r'gen\s+ai', r'genai',
    r'large\s+language\s+model', r'\bllm\b', r'\bllms\b',
    r'openai', r'deepmind', r'copilot', r'midjourney',

    # Chatbots & Conversational AI
    r'chatbot', r'virtual\s+assistant',

    # Computer vision & NLP sub-fields
    r'facial\s+recognition', r'computer\s+vision',
    r'natural\s+language\s+processing', r'\bnlp\b',

    # Autonomous systems
    r'autonomous\s+vehicle', r'self.driving',

    # Ethics, Risk & Society
    r'algorithmic\s+bias', r'ai\s+ethics', r'ai\s+bias',
    r'deepfake',
    r'existential\s+risk', r'autonomous\s+weapon',
    r'predictive\s+policing', r'recommendation\s+algorithm',
    r'data\s+mining',
]

KEYWORD_REGEX = re.compile(
    r'\b(?:' + '|'.join(KEYWORDS) + r')\b',
    re.IGNORECASE
)


def standardize_phrases(text_str):
    """Merge key multi-word AI phrases into underscore-joined tokens."""
    if not isinstance(text_str, str):
        return ""
    s = text_str.lower()
    s = re.sub(r'\bartificial\s+intelligence\b',       'artificial_intelligence',        s)
    s = re.sub(r'\bmachine\s+learning\b',               'machine_learning',               s)
    s = re.sub(r'\bdeep\s+learning\b',                  'deep_learning',                  s)
    s = re.sub(r'\bneural\s+networks?\b',               'neural_network',                 s)
    s = re.sub(r'\blarge\s+language\s+models?\b',       'large_language_model',           s)
    s = re.sub(r'\bgenerative\s+ai\b',                  'generative_ai',                  s)
    s = re.sub(r'\bfacial\s+recognition\b',             'facial_recognition',             s)
    s = re.sub(r'\bself[- ]driving\b',                  'self_driving',                   s)
    s = re.sub(r'\bautonomous\s+vehicles?\b',           'autonomous_vehicle',             s)
    s = re.sub(r'\bcomputer\s+vision\b',                'computer_vision',                s)
    s = re.sub(r'\bnatural\s+language\s+processing\b',  'natural_language_processing',    s)
    s = re.sub(r'\bpredictive\s+policing\b',            'predictive_policing',            s)
    s = re.sub(r'\bdeep\s*fakes?\b',                    'deepfake',                       s)
    s = re.sub(r'\bvirtual\s+assistant[s]?\b',          'virtual_assistant',              s)
    s = re.sub(r'\bsam\s+altman\b',                     'sam_altman',                     s)
    s = re.sub(r'\bsilicon\s+valley\b',                 'silicon_valley',                 s)
    s = re.sub(r'\bdata\s+mining\b',                    'data_mining',                    s)
    s = re.sub(r'\balgorithmic\s+bias\b',               'algorithmic_bias',               s)
    s = re.sub(r'\bai\s+ethics\b',                      'ai_ethics',                      s)
    return s


def clean_headline(val):
    """Extract the 'main' headline string from NYT dict-encoded headline fields."""
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.startswith('{'):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, dict) and 'main' in parsed:
                return str(parsed['main'])
        except Exception:
            pass
    return val_str

# ═════════════════════════════════════════════════════════════════════════════
# END mirrored block
# ═════════════════════════════════════════════════════════════════════════════


def assign_epoch(year):
    for label, lo, hi in EPOCHS:
        if lo <= year <= hi:
            return label
    return "Unknown"


def build_corpus(filepath):
    """Stream the NYT CSV, apply the RQ1 filter, and keep RQ2-relevant columns
    plus a precise New-York-timezone timestamp."""
    print(f"[*] Building RQ2 corpus from: {filepath}")
    filtered_chunks = []
    chunk_iterator = pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk in chunk_iterator:
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        if 'pub_date' not in chunk.columns or 'abstract' not in chunk.columns:
            continue

        # Year exactly as RQ1 (first 4 chars of the UTC pub_date string) so the
        # corpus membership and epoch assignment match RQ1 precisely.
        chunk['year'] = chunk['pub_date'].astype(str).str[:4]
        chunk = chunk[chunk['year'].str.isnumeric()].copy()
        chunk['year'] = chunk['year'].astype(int)
        chunk = chunk[(chunk['year'] >= 2000) & (chunk['year'] <= 2025)]
        if chunk.empty:
            continue

        if 'type_of_material' in chunk.columns:
            chunk = chunk[~chunk['type_of_material'].isin(EXCLUDE_MATERIALS)]
        if 'section_name' in chunk.columns:
            chunk = chunk[
                ~chunk['section_name'].astype(str).str.strip().str.title()
                .isin(EXCLUDE_SECTIONS)
            ]
        if chunk.empty:
            continue

        headline_col = 'headline' if 'headline' in chunk.columns else chunk.columns[0]
        chunk['clean_headline'] = chunk[headline_col].apply(clean_headline)
        chunk['abstract'] = chunk['abstract'].fillna('')

        # Same keyword match as RQ1: standardize phrases first, then regex.
        match_text = (chunk['clean_headline'] + " " + chunk['abstract']).apply(standardize_phrases)
        matched = chunk[match_text.str.contains(KEYWORD_REGEX, na=False)].copy()
        if matched.empty:
            continue

        # --- RQ2 enrichment: precise New-York-timezone timestamp ------------- #
        # Raw pub_date is UTC ('...+00:00'); parse as UTC then convert to NY.
        dt_utc = pd.to_datetime(matched['pub_date'], utc=True, errors='coerce')
        matched['pub_datetime_ny'] = dt_utc.dt.tz_convert(NY_TZ)
        matched = matched[matched['pub_datetime_ny'].notna()]

        filtered_chunks.append(
            matched[['pub_datetime_ny', 'year', 'clean_headline', 'abstract']]
        )

    if not filtered_chunks:
        raise ValueError("No matching records found. Check keyword filter and file path.")

    df = pd.concat(filtered_chunks, ignore_index=True)

    # Derived metadata (deterministic from the date) -------------------------- #
    df['epoch'] = df['year'].apply(assign_epoch)
    df['is_post_chatgpt'] = df['pub_datetime_ny'] >= CHATGPT_RELEASE_NY

    return df


def print_summary(df):
    print("\n" + "=" * 70)
    print("RQ2 CORPUS SUMMARY")
    print("=" * 70)
    print(f"Total AI-related NYT articles: {len(df):,}")
    print(f"Date range (NY): {df['pub_datetime_ny'].min()}  ->  {df['pub_datetime_ny'].max()}")

    print("\nPer-epoch counts:")
    for label, lo, hi in EPOCHS:
        print(f"  {label} ({lo}-{hi}): {int((df['epoch'] == label).sum()):,}")

    era3 = df[df['epoch'] == 'Era3']
    print("\nEra3 internal ChatGPT split (NY date 2022-11-30):")
    print(f"  pre-ChatGPT  (2019-01-01 .. 2022-11-29): {int((~era3['is_post_chatgpt']).sum()):,}")
    print(f"  post-ChatGPT (2022-11-30 .. 2025):       {int(era3['is_post_chatgpt'].sum()):,}")


def main():
    if not os.path.exists(DATA_FILE):
        print(f"[!] '{DATA_FILE}' not found.")
        print("    Place the Kaggle 'nyt-metadata.csv' in the project root.")
        return

    df = build_corpus(DATA_FILE)
    print_summary(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[+] Saved RQ2 corpus -> '{OUTPUT_FILE}'  ({len(df):,} rows)")


if __name__ == "__main__":
    main()

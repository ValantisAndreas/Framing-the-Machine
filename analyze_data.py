"""
analyze_data.py
Framing the Machine — RQ1: Thematic Evolution of AI Coverage in the NYT (2000–2025)

Methodology: Epoch-stratified LDA with Hungarian topic alignment.
             Gensim c_v coherence sweep selects the optimal topic count (k) automatically.

Workflow:
  1. Run the script once → inspect the TOPIC LABELING GUIDE printed at the end.
  2. Fill in the TOPIC_LABELS dict below with meaningful names (e.g. "Labor & Automation").
  3. Re-run → all charts and JSON output will use your labels.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import ast
import re
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE = "nyt-metadata.csv"

# Coherence sweep: evaluate k topics from K_MIN to K_MAX (inclusive).
# The script picks the k with the highest c_v score automatically.
# NOTE: K_MAX is capped at 6.
K_MIN = 3
K_MAX = 6
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TOPIC LABELS  (fill in after first run)
#
# After the script prints the TOPIC LABELING GUIDE, read the top words per
# track and assign a short descriptive label for each topic index below.
# Leave as empty string "" to fall back to "Topic N" in charts and JSON.
#
# Example:
#   TOPIC_LABELS = {
#       0: "Scientific Research & Academia",
#       1: "Labor Displacement & Automation",
#       2: "GenAI Platforms & Products",
#       3: "Ethics, Surveillance & Society",
#       4: "Robotics & Autonomous Systems",
#   }
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_LABELS = {
    0: "Scientific Computing & Emerging AI Platforms",
    1: "Autonomous Vehicles & Transportation AI",
    2: "Labor Automation & Workforce Disruption",
    3: "Consumer Robotics to GenAI & Synthetic Media",
    4: "AI Safety, Policy & Corporate Leadership",
    # Slots 5-9 unused at k=5 
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
}

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

    # Ethics, Risk & Society  ← primary expansion for RQ1 risk-framing coverage
    r'algorithmic\s+bias', r'ai\s+ethics', r'ai\s+bias',
    # NOTE: 'surveillance' is intentionally EXCLUDED as a standalone keyword.
    # In NYT coverage 2000-2025, 'surveillance' predominantly refers to NSA/
    # Snowden/wiretapping/police body-cam journalism — not AI. Coherence scores
    # confirm this: the clean 1,253-doc corpus (without surveillance) scores
    # 0.40-0.45 vs 0.29-0.33 with it. AI surveillance articles are captured
    # anyway via 'facial_recognition', 'predictive_policing', 'deepfake' etc.
    r'deepfake',
    r'existential\s+risk', r'autonomous\s+weapon',
    r'predictive\s+policing', r'recommendation\s+algorithm',
    r'data\s+mining',
]

KEYWORD_REGEX = re.compile(
    r'\b(?:' + '|'.join(KEYWORDS) + r')\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
#
# Two categories of stop words are merged:
#   (a) sklearn's standard English stop words
#   (b) Domain-ubiquitous or noisy words that cause topic collapse.
#       Key additions vs. the original: robot, robots, technology, tech,
#       company, companies, data, use/used/using, says/say, help, start,
#       york (NYT artefact), web, com, article, photos, percent, billion,
#       million, chief, jay, abb — all previously caused every topic to
#       share the same high-weight centroid words.
# ─────────────────────────────────────────────────────────────────────────────
NOISE_STOP_WORDS = {
    # Previously dominated ALL topic tracks → removed
    'robot', 'robots', 'technology', 'tech', 'data',
    'use', 'used', 'using', 'company', 'companies',
    'system', 'systems', 'information',
    # Generic reporting / filler verbs
    'says', 'say', 'said', 'help', 'helps', 'start', 'starts',
    # NYT-specific CSV / metadata artefacts
    'york', 'new', 'times', 'com', 'article', 'articles',
    'photos', 'photo', 'page', 'pages', 'web', 'online',
    # Generic business / numeric noise
    'percent', 'billion', 'million', 'chief', 'board',
    'report', 'reports', 'news',
    # Proper noun noise from early corpus
    'jay', 'abb', 'tokyo', 'farhad',
    # NYT newsletter/digest artifacts (Scuttlebot was NYT's tech roundup column)
    'scuttlebot', 'scour', 'peculiar', 'editors', 'reporters', 'items', 'selection', 'includes',
    # Political figures / events — caught by 'surveillance' keyword but not AI-specific
    'obama', 'bush', 'iraq', 'laden', 'bin', 'terrorist',
    'president', 'officials', 'senate', 'congress', 'federal', 'administration',
    # Security/law terms that dominate surveillance articles but carry no AI signal
    'court', 'law', 'police', 'program', 'government', 'national', 'security',
    'nsa', 'cia', 'fbi', 'agency', 'intelligence', 'warrant', 'surveillance',
    # Note: 'surveillance' added here too so it doesn't dominate topic word lists
    # Standalone parts of standardised phrases (prevent double-counting)
    'artificial', 'intelligence', 'machine', 'learning',
    'neural', 'network', 'networks',
    'large', 'language', 'model', 'models',
    'silicon', 'valley', 'sam', 'altman',
    # Generic temporal / locative noise
    'street', 'west', 'east', 'north', 'south', 'broadway',
    'theater', 'avenue', 'mr', 'mrs', 'ms', 'dr',
    'today', 'yesterday', 'tomorrow', 'tonight',
    'night', 'day', 'week', 'year', 'years', 'month', 'months',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday',
    'c1', 'b1', 'a1', 'd1',
    'world', 'people', 'make', 'way', 'called', 'like', 'just', 'time',
}

custom_stop_words = set(sk_text.ENGLISH_STOP_WORDS).union(NOISE_STOP_WORDS)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

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


def load_and_filter_data(filepath):
    """Stream the NYT CSV in chunks, filter to AI-related articles (2000–2025)."""
    print("[*] Loading and filtering data (expanded keyword set)...")
    filtered_chunks = []
    chunk_iterator = pd.read_csv(filepath, chunksize=50_000, low_memory=False)

    exclude_materials = {
        'Schedule', 'Paid Death Notice', 'Obituary', 'Letter',
        'Review', 'List', 'Summary', 'Brief'
    }
    exclude_sections = {
        'Sports', 'Sports Desk', 'Crosswords', 'Arts',
        'Theater', 'Movies', 'Travel', 'Real Estate'
    }

    for chunk in chunk_iterator:
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        if 'pub_date' not in chunk.columns or 'abstract' not in chunk.columns:
            continue

        chunk['year'] = chunk['pub_date'].astype(str).str[:4]
        chunk = chunk[chunk['year'].str.isnumeric()].copy()
        chunk['year'] = chunk['year'].astype(int)
        chunk = chunk[(chunk['year'] >= 2000) & (chunk['year'] <= 2025)]
        if chunk.empty:
            continue

        if 'type_of_material' in chunk.columns:
            chunk = chunk[~chunk['type_of_material'].isin(exclude_materials)]
        if 'section_name' in chunk.columns:
            chunk = chunk[
                ~chunk['section_name'].astype(str).str.strip().str.title()
                .isin(exclude_sections)
            ]
        if chunk.empty:
            continue

        headline_col = 'headline' if 'headline' in chunk.columns else chunk.columns[0]
        chunk['clean_headline'] = chunk[headline_col].apply(clean_headline)
        chunk['abstract'] = chunk['abstract'].fillna('')
        chunk['combined_text'] = (
            chunk['clean_headline'] + " " + chunk['abstract']
        ).apply(standardize_phrases)

        matched = chunk[
            chunk['combined_text'].str.contains(KEYWORD_REGEX, na=False)
        ].copy()

        if not matched.empty:
            filtered_chunks.append(matched[['year', 'combined_text']])

    if not filtered_chunks:
        raise ValueError("No matching records found. Check keyword filter and file path.")

    df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"[+] Extracted {len(df):,} AI-related articles.")
    return df


def get_label(topic_id):
    """Return a display label, falling back to 'Topic N' if not set."""
    lbl = TOPIC_LABELS.get(topic_id, "")
    return lbl if lbl else f"Topic {topic_id}"


def run_coherence_sweep(tokenized_docs, k_min, k_max):
    """
    Train a Gensim LDA model for each k in [k_min, k_max] and compute the
    c_v coherence score.  Returns (optimal_k, {k: score}) for plotting.

    c_v coherence measures the semantic similarity of top words within each
    topic using a sliding-window PMI approach — higher is better.
    """
    print(f"\n[*] Coherence sweep: evaluating k = {k_min} → {k_max} ...")

    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    scores = {}
    for k in range(k_min, k_max + 1):
        lda_g = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=10,
            workers=2,
            per_word_topics=False,
        )
        cm = CoherenceModel(
            model=lda_g,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v',
        )
        score = cm.get_coherence()
        scores[k] = round(score, 4)
        print(f"    k = {k:2d}   c_v coherence = {score:.4f}")

    optimal_k = max(scores, key=scores.get)
    print(f"\n[+] Optimal k = {optimal_k}  (coherence = {scores[optimal_k]:.4f})")
    return optimal_k, scores


def plot_coherence_sweep(scores, optimal_k, out_path):
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, vals, marker='o', linewidth=2, markersize=8, color='steelblue', label='c_v coherence')
    ax.axvline(x=optimal_k, color='tomato', linestyle='--', linewidth=1.8,
               label=f'Optimal k = {optimal_k}')
    ax.fill_between(ks, vals, alpha=0.08, color='steelblue')
    ax.set_title("LDA Topic Coherence Sweep (c_v) — Selecting Optimal k", fontsize=14, weight='bold')
    ax.set_xlabel("Number of Topics  k", fontsize=12)
    ax.set_ylabel("Coherence Score  (c_v, higher = better)", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    [+] Saved coherence sweep plot → '{out_path}'")



def print_labeling_guide(n_topics, feature_names, comp1, comp2, comp3,
                         mapping_1_to_2, mapping_1_to_3):
    """
    Print the top-10 words for every aligned topic track across all three eras.
    Use this output to fill in TOPIC_LABELS at the top of the file.
    """
    all_labeled = all(TOPIC_LABELS.get(t, "") for t in range(n_topics))
    status = "✓ All topics labeled." if all_labeled else "⚠  Unlabeled topics — fill in TOPIC_LABELS and re-run."

    print("\n" + "=" * 80)
    print("TOPIC LABELING GUIDE")
    print(status)
    print("=" * 80)
    print("Read the top words for each track across all three eras, then assign")
    print("a short thematic label in the TOPIC_LABELS dict at the top of this file.\n")

    for t_ref in range(n_topics):
        t2 = mapping_1_to_2[t_ref]
        t3 = mapping_1_to_3[t_ref]

        top1 = [feature_names[i] for i in comp1[t_ref].argsort()[:-11:-1]]
        top2 = [feature_names[i] for i in comp2[t2].argsort()[:-11:-1]]
        top3 = [feature_names[i] for i in comp3[t3].argsort()[:-11:-1]]

        current = TOPIC_LABELS.get(t_ref, "")
        lbl_str = f'"{current}"' if current else "*** NOT YET LABELED ***"

        print(f"  TOPIC_LABELS[{t_ref}] = {lbl_str}")
        print(f"    Era 1 (2000–2011): {', '.join(top1)}")
        print(f"    Era 2 (2012–2018): {', '.join(top2)}")
        print(f"    Era 3 (2019–2025): {', '.join(top3)}")
        print()

    if not all_labeled:
        print("─" * 80)
        print("ACTION REQUIRED: Edit TOPIC_LABELS in analyze_data.py and re-run.")
        print("─" * 80)



def main():
    if not os.path.exists(DATA_FILE):
        print(f"[!] '{DATA_FILE}' not found.")
        return

    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    df_ai = load_and_filter_data(DATA_FILE)

    df_ai['epoch'] = np.select(
        [df_ai['year'] <= 2011,
         (df_ai['year'] >= 2012) & (df_ai['year'] <= 2018),
         df_ai['year'] >= 2019],
        ['Era1', 'Era2', 'Era3'],
        default='Unknown'
    )

    # ── 2. Vectorize (sklearn) ────────────────────────────────────────────────
    print("\n[*] Vectorizing corpus ...")
    vectorizer = CountVectorizer(
        stop_words=list(custom_stop_words),
        token_pattern=r"(?u)\b[a-zA-Z_]{3,}\b",
        ngram_range=(1, 1),
        max_df=0.95,
        min_df=5,           # Raised from 3 — filters rare noisy tokens
    )
    dtm = vectorizer.fit_transform(df_ai['combined_text'])
    feature_names = vectorizer.get_feature_names_out()
    print(f"[+] Vocabulary size: {len(feature_names):,} tokens")

    # ── 3. Coherence sweep (gensim) to pick N_TOPICS ──────────────────────────
    # Tokenise into the same vocabulary as sklearn for consistency
    vocab_set = set(feature_names)
    tokenized_docs = [
        [w for w in doc.split() if w in vocab_set]
        for doc in df_ai['combined_text']
    ]
    tokenized_docs = [d for d in tokenized_docs if d]  # drop empty docs

    n_topics, coherence_scores = run_coherence_sweep(tokenized_docs, K_MIN, K_MAX)
    plot_coherence_sweep(coherence_scores, n_topics, "results/coherence_sweep.png")

    # ── 4. Split per epoch ────────────────────────────────────────────────────
    mask1 = (df_ai['epoch'] == 'Era1').values
    mask2 = (df_ai['epoch'] == 'Era2').values
    mask3 = (df_ai['epoch'] == 'Era3').values

    dtm_era1, dtm_era2, dtm_era3 = dtm[mask1], dtm[mask2], dtm[mask3]
    print(f"\n[*] Era 1 (2000–2011): {dtm_era1.shape[0]:,} docs")
    print(f"    Era 2 (2012–2018): {dtm_era2.shape[0]:,} docs")
    print(f"    Era 3 (2019–2025): {dtm_era3.shape[0]:,} docs")

    # ── 5. Train epoch-stratified LDA ─────────────────────────────────────────
    print(f"\n[*] Training epoch-stratified LDA  (k={n_topics}, max_iter=20) ...")
    lda_kw = dict(n_components=n_topics, random_state=RANDOM_STATE,
                  max_iter=20, learning_method='batch')
    lda1 = LatentDirichletAllocation(**lda_kw).fit(dtm_era1)
    lda2 = LatentDirichletAllocation(**lda_kw).fit(dtm_era2)
    lda3 = LatentDirichletAllocation(**lda_kw).fit(dtm_era3)

    # Normalise to probability distributions
    def norm(comp):
        return comp / comp.sum(axis=1, keepdims=True)

    comp1, comp2, comp3 = norm(lda1.components_), norm(lda2.components_), norm(lda3.components_)

    # ── 6. Hungarian alignment (Era 1 as baseline) ───────────────────────────
    def hungarian_align(base, target):
        sim = cosine_similarity(base, target)
        rows, cols = linear_sum_assignment(1.0 - sim)
        return {int(r): int(c) for r, c in zip(rows, cols)}

    mapping_1_to_2 = hungarian_align(comp1, comp2)
    mapping_1_to_3 = hungarian_align(comp1, comp3)

    # ── 7. Topic labeling guide ───────────────────────────────────────────────
    print_labeling_guide(n_topics, feature_names,
                         comp1, comp2, comp3,
                         mapping_1_to_2, mapping_1_to_3)

    # ── 8. Document-topic distributions ──────────────────────────────────────
    print("[*] Computing document-topic distributions ...")
    w1 = lda1.transform(dtm_era1)
    w2 = lda2.transform(dtm_era2)
    w3 = lda3.transform(dtm_era3)

    # Re-align Era 2/3 columns back to Era 1 coordinates
    aw2 = np.zeros_like(w2)
    aw3 = np.zeros_like(w3)
    for t in range(n_topics):
        aw2[:, t] = w2[:, mapping_1_to_2[t]]
        aw3[:, t] = w3[:, mapping_1_to_3[t]]

    mw1, mw2, mw3 = w1.mean(axis=0), aw2.mean(axis=0), aw3.mean(axis=0)

    elapsed = time.time() - t0
    print(f"\n[+] Completed in {elapsed:.1f}s")

    # ── 9. Save JSON ──────────────────────────────────────────────────────────
    topic_tracks = []
    for t in range(n_topics):
        t2, t3 = mapping_1_to_2[t], mapping_1_to_3[t]
        top1 = comp1[t].argsort()[:-11:-1]
        top2 = comp2[t2].argsort()[:-11:-1]
        top3 = comp3[t3].argsort()[:-11:-1]
        topic_tracks.append({
            'track_id': t,
            'label': get_label(t),
            'era1_top_words': [{'word': feature_names[i], 'weight': float(comp1[t][i])} for i in top1],
            'era2_top_words': [{'word': feature_names[i], 'weight': float(comp2[t2][i])} for i in top2],
            'era3_top_words': [{'word': feature_names[i], 'weight': float(comp3[t3][i])} for i in top3],
        })

    out = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time_seconds': round(elapsed, 2),
        'total_documents': len(df_ai),
        'optimal_n_topics': int(n_topics),
        'coherence_scores_cv': {str(k): v for k, v in coherence_scores.items()},
        'era_document_counts': {
            'Era1_2000_2011': int(dtm_era1.shape[0]),
            'Era2_2012_2018': int(dtm_era2.shape[0]),
            'Era3_2019_2025': int(dtm_era3.shape[0]),
        },
        'macro_topic_prevalences': [
            {
                'track_id': t,
                'label': get_label(t),
                'prevalence_era1': float(mw1[t]),
                'prevalence_era2': float(mw2[t]),
                'prevalence_era3': float(mw3[t]),
            }
            for t in range(n_topics)
        ],
        'topic_tracks': topic_tracks,
    }

    json_path = "results/aligned_lda_results.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"    [+] Saved JSON → '{json_path}'")

    # ── 10. Visualisations ───────────────────────────────────────────────────
    print("[*] Generating charts ...")
    sns.set_theme(style="whitegrid")

    # Chart A — Macro topic prevalence drift
    rows = []
    era_labels = ['Era 1\n(2000–2011)', 'Era 2\n(2012–2018)', 'Era 3\n(2019–2025)']
    for t in range(n_topics):
        for era_lbl, weight in zip(era_labels, [float(mw1[t]), float(mw2[t]), float(mw3[t])]):
            rows.append({'Topic': get_label(t), 'Mean Prevalence': weight, 'Era': era_lbl})

    df_prev = pd.DataFrame(rows)
    fig_w = max(10, n_topics * 2)
    plt.figure(figsize=(fig_w, 6))
    sns.barplot(data=df_prev, x='Topic', y='Mean Prevalence', hue='Era', palette='muted')
    plt.title(
        f"Macro-Level Topic Prevalence Drift Across Three Tech Epochs (RQ1)\n"
        f"[k={n_topics}, selected by c_v coherence sweep]",
        fontsize=13, weight='bold'
    )
    plt.ylabel("Mean Topic Weight in Documents", fontsize=11)
    plt.xlabel("Topic Track", fontsize=11)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.legend(title="Tech Epoch")
    plt.tight_layout()
    plt.savefig("results/aligned_topic_prevalence.png", dpi=300)
    plt.close()
    print("    [+] Saved topic prevalence chart.")

    # Chart B — Key term weight shifts per topic track
    fig, axes = plt.subplots(n_topics, 1, figsize=(13, 5 * n_topics))
    if n_topics == 1:
        axes = [axes]

    for t in range(n_topics):
        t2, t3 = mapping_1_to_2[t], mapping_1_to_3[t]
        # Use Era 3's top 5 words as the basis for cross-era comparison
        top5_idx3 = comp3[t3].argsort()[:-6:-1]
        key_words = [feature_names[i] for i in top5_idx3]

        plot_rows = []
        for word in key_words:
            idx = vectorizer.vocabulary_.get(word)
            if idx is None:
                continue
            plot_rows.append({'Word': word, 'Weight': float(comp1[t][idx]),   'Era': 'Era 1 (2000–2011)'})
            plot_rows.append({'Word': word, 'Weight': float(comp2[t2][idx]),  'Era': 'Era 2 (2012–2018)'})
            plot_rows.append({'Word': word, 'Weight': float(comp3[t3][idx]),  'Era': 'Era 3 (2019–2025)'})

        df_plot = pd.DataFrame(plot_rows)
        ax = axes[t]
        if not df_plot.empty:
            sns.barplot(data=df_plot, x='Word', y='Weight', hue='Era', ax=ax, palette='muted')
        ax.set_title(
            f"Track {t}: {get_label(t)} — Era 3 Key Term Weights Across Epochs",
            fontsize=11, weight='bold'
        )
        ax.set_ylabel("Word Probability")
        ax.set_xlabel("")
        if t == 0:
            ax.legend(title="Tech Epoch", fontsize=9)
        else:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    plt.suptitle(
        "Linguistic Evolution of Key AI Terms (2000–2025)",
        fontsize=15, weight='bold', y=1.005
    )
    plt.tight_layout()
    plt.savefig("results/aligned_topics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("    [+] Saved key-term comparison chart.")

    print("\n✓ All outputs saved to results/")
    print("  → Next step: fill in TOPIC_LABELS at the top of analyze_data.py and re-run.\n")


if __name__ == "__main__":
    main()

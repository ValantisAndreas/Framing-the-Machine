import pandas as pd
import numpy as np
import os
import ast
import re
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# Define Configuration Constants
DATA_FILE = "nyt-metadata.csv"
N_TOPICS = 5
RANDOM_STATE = 42

KEYWORDS = [
    # Core AI & ML
    r'artificial\s+intelligence', r'machine\s+learning', r'deep\s+learning', 
    r'neural\s+network', r'algorithm',
    
    # Robotics & Labor
    r'automation', r'robot',
    
    # Generative AI & Platforms (Modern Era)
    r'chatgpt', r'generative\s+ai', r'gen\s+ai', r'genai',
    r'large\s+language\s+model', r'llm', r'llms',
    r'openai', r'deepmind', r'copilot', r'midjourney'
]

KEYWORD_REGEX = re.compile(r'\b(?:' + '|'.join(KEYWORDS) + r')\b', re.IGNORECASE)

# Text Preprocessing & Cleaning setup
from sklearn.feature_extraction import text
standard_stop_words = set(text.ENGLISH_STOP_WORDS)
custom_stop_words = standard_stop_words.union({
    'street', 'west', 'east', 'north', 'south', 'broadway', 'theater', 'theaters', 'avenue',
    'mr', 'mrs', 'ms', 'dr', 'said', 'new', 'today', 'yesterday', 'tomorrow', 'tonight',
    'night', 'day', 'week', 'year', 'years', 'month', 'months', 'monday', 'tuesday', 'wednesday',
    'thursday', 'friday', 'saturday', 'sunday', 'photo', 'page', 'pages', 'c1', 'b1', 'a1', 'd1',
    'like', 'just', 'time', 'work', 'world', 'people', 'make', 'way', 'times', 'called',
    # Standalone components of standardized phrases (filtered out to prevent redundancy)
    'artificial', 'intelligence', 'learning', 'machine', 'valley', 'silicon', 'sam', 'altman',
    'large', 'language', 'model', 'models', 'neural', 'network', 'networks'
})

def standardize_phrases(text_str):
    if not isinstance(text_str, str):
        return ""
    text_str = text_str.lower()
    # Merge key multi-word phrases into cohesive tokens with underscores
    text_str = re.sub(r'\bartificial\s+intelligence\b', 'artificial_intelligence', text_str)
    text_str = re.sub(r'\bmachine\s+learning\b', 'machine_learning', text_str)
    text_str = re.sub(r'\bdeep\s+learning\b', 'deep_learning', text_str)
    text_str = re.sub(r'\bneural\s+networks?\b', 'neural_network', text_str)
    text_str = re.sub(r'\blarge\s+language\s+models?\b', 'large_language_model', text_str)
    text_str = re.sub(r'\bgenerative\s+ai\b', 'generative_ai', text_str)
    text_str = re.sub(r'\bsam\s+altman\b', 'sam_altman', text_str)
    text_str = re.sub(r'\bsilicon\s+valley\b', 'silicon_valley', text_str)
    return text_str

def clean_headline(val):
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.startswith('{'):
        try:
            parsed_dict = ast.literal_eval(val_str)
            if isinstance(parsed_dict, dict) and 'main' in parsed_dict:
                return str(parsed_dict['main'])
        except Exception:
            pass
    return val_str

def load_and_filter_data(filepath):
    print(f"[*] Preprocessing data...")
    filtered_chunks = []
    chunk_iterator = pd.read_csv(filepath, chunksize=50000, low_memory=False)
    
    # Exclude non-news formats
    exclude_materials = {'Schedule', 'Paid Death Notice', 'Obituary', 'Letter', 'Review', 'List', 'Summary', 'Brief'}
    exclude_sections = {'Sports', 'Sports Desk', 'Crosswords', 'Arts', 'Theater', 'Movies', 'Travel', 'Real Estate'}
    
    for i, chunk in enumerate(chunk_iterator):
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
            chunk = chunk[~chunk['section_name'].astype(str).str.strip().str.title().isin(exclude_sections)]
            
        if chunk.empty:
            continue
            
        headline_col = 'headline' if 'headline' in chunk.columns else chunk.columns[0]
        chunk['clean_headline'] = chunk[headline_col].apply(clean_headline)
        chunk['abstract'] = chunk['abstract'].fillna('')
        chunk['combined_text'] = (chunk['clean_headline'] + " " + chunk['abstract']).apply(standardize_phrases)
        
        is_ai_related = chunk['combined_text'].str.contains(KEYWORD_REGEX, na=False)
        matched_rows = chunk[is_ai_related].copy()
        
        if not matched_rows.empty:
            filtered_chunks.append(matched_rows[['year', 'combined_text']])
            
    if not filtered_chunks:
        raise ValueError("No matching records found.")
        
    final_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"[+] Extracted {len(final_df)} filtered articles.")
    return final_df

def main():
    if not os.path.exists(DATA_FILE):
        print(f"[!] Error: '{DATA_FILE}' not found in the root directory.")
        return

    # Initialize results directory
    os.makedirs("results", exist_ok=True)
    
    # Start timer instrumentation
    start_time = time.time()
    
    # Load and clean data
    df_ai = load_and_filter_data(DATA_FILE)
    
    # Assign technological epochs
    conditions = [
        (df_ai['year'] <= 2011),
        (df_ai['year'] >= 2012) & (df_ai['year'] <= 2018),
        (df_ai['year'] >= 2019)
    ]
    choices = ['Era1', 'Era2', 'Era3']
    df_ai['epoch'] = np.select(conditions, choices, default='Unknown')
    
    # Vectorize text over entire vocabulary
    print("[*] Vectorizing text data (excluding numbers and custom stop words)...")
    vectorizer = CountVectorizer(
        stop_words=list(custom_stop_words),
        token_pattern=r"(?u)\b[a-zA-Z_]{3,}\b",
        ngram_range=(1,1),
        max_df=0.95,
        min_df=3
    )
    dtm = vectorizer.fit_transform(df_ai['combined_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Split DTMs per Epoch
    dtm_era1 = dtm[(df_ai['epoch'] == 'Era1').values]
    dtm_era2 = dtm[(df_ai['epoch'] == 'Era2').values]
    dtm_era3 = dtm[(df_ai['epoch'] == 'Era3').values]
    
    print(f"[*] Pre-DL Era docs: {dtm_era1.shape[0]}")
    print(f"[*] DL Expansion Era docs: {dtm_era2.shape[0]}")
    print(f"[*] GenAI Integration Era docs: {dtm_era3.shape[0]}")
    
    # Train independent LDA models (no temporal leakage)
    print(f"[*] Training independent LDA model for Era 1 (2000-2011)...")
    lda1 = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=10)
    lda1.fit(dtm_era1)
    
    print(f"[*] Training independent LDA model for Era 2 (2012-2018)...")
    lda2 = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=10)
    lda2.fit(dtm_era2)
    
    print(f"[*] Training independent LDA model for Era 3 (2019-2025)...")
    lda3 = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=10)
    lda3.fit(dtm_era3)
    
    # Normalize components to represent probability distributions
    comp1 = lda1.components_ / lda1.components_.sum(axis=1, keepdims=True)
    comp2 = lda2.components_ / lda2.components_.sum(axis=1, keepdims=True)
    comp3 = lda3.components_ / lda3.components_.sum(axis=1, keepdims=True)
    
    # Align Topics using Cosine Similarity & Hungarian Algorithm
    # Align Era 2 to Era 1 (Baseline)
    sim_1_2 = cosine_similarity(comp1, comp2)
    row_ind2, col_ind2 = linear_sum_assignment(1.0 - sim_1_2)
    mapping_1_to_2 = {r: c for r, c in zip(row_ind2, col_ind2)}
    
    # Align Era 3 to Era 1 (Baseline)
    sim_1_3 = cosine_similarity(comp1, comp3)
    row_ind3, col_ind3 = linear_sum_assignment(1.0 - sim_1_3)
    mapping_1_to_3 = {r: c for r, c in zip(row_ind3, col_ind3)}
    
    print("\n" + "="*80)
    print("ALIGNED TEMPORAL LDA")
    print("="*80)
    
    json_topic_tracks = []
    
    for t_ref in range(N_TOPICS):
        t_era2 = mapping_1_to_2[t_ref]
        t_era3 = mapping_1_to_3[t_ref]
        
        # Get top words and exact probability weights for each aligned topic
        top_indices1 = comp1[t_ref].argsort()[:-11:-1]
        words1 = [feature_names[i] for i in top_indices1]
        weights1 = [float(comp1[t_ref][i]) for i in top_indices1]
        
        top_indices2 = comp2[t_era2].argsort()[:-11:-1]
        words2 = [feature_names[i] for i in top_indices2]
        weights2 = [float(comp2[t_era2][i]) for i in top_indices2]
        
        top_indices3 = comp3[t_era3].argsort()[:-11:-1]
        words3 = [feature_names[i] for i in top_indices3]
        weights3 = [float(comp3[t_era3][i]) for i in top_indices3]
        
        print("\n" + "-"*90)
        print(f"ALIGNED TOPIC TRACK {t_ref} (Era 1: Topic {t_ref} -> Era 2: Topic {t_era2} -> Era 3: Topic {t_era3})")
        print("-"*90)
        
        df_drift = pd.DataFrame({
            'Era 1 Word': words1,
            'Era 1 Weight': [f"{w:.4f}" for w in weights1],
            'Era 2 Word': words2,
            'Era 2 Weight': [f"{w:.4f}" for w in weights2],
            'Era 3 Word': words3,
            'Era 3 Weight': [f"{w:.4f}" for w in weights3]
        })
        print(df_drift.to_string(index=False))
        
        # Save track statistics for JSON export
        json_topic_tracks.append({
            'track_id': t_ref,
            'era1_topic_id': t_ref,
            'era2_topic_id': int(t_era2),
            'era3_topic_id': int(t_era3),
            'era1_top_words': [{'word': w, 'weight': wt} for w, wt in zip(words1, weights1)],
            'era2_top_words': [{'word': w, 'weight': wt} for w, wt in zip(words2, weights2)],
            'era3_top_words': [{'word': w, 'weight': wt} for w, wt in zip(words3, weights3)]
        })
        
    # Calculate document-topic distributions per epoch and align them
    print("[*] Calculating and aligning document-topic distributions for macro analysis...")
    weights_era1 = lda1.transform(dtm_era1)
    weights_era2 = lda2.transform(dtm_era2)
    weights_era3 = lda3.transform(dtm_era3)
    
    # Align Era 2 and Era 3 topic probability columns back to Era 1 baseline coordinates
    aligned_weights_era2 = np.zeros_like(weights_era2)
    aligned_weights_era3 = np.zeros_like(weights_era3)
    for t_ref in range(N_TOPICS):
        aligned_weights_era2[:, t_ref] = weights_era2[:, mapping_1_to_2[t_ref]]
        aligned_weights_era3[:, t_ref] = weights_era3[:, mapping_1_to_3[t_ref]]
        
    # Take the mean of each aligned topic coordinate to measure overall prevalence per epoch
    mean_weights1 = weights_era1.mean(axis=0)
    mean_weights2 = aligned_weights_era2.mean(axis=0)
    mean_weights3 = aligned_weights_era3.mean(axis=0)
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    print(f"\n[+] Processing completed in {elapsed_time:.2f} seconds.")
    
    # Save detailed JSON metrics
    results_json = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time_seconds': float(f"{elapsed_time:.2f}"),
        'total_documents': len(df_ai),
        'era_document_counts': {
            'Era 1 (2000-2011)': int(dtm_era1.shape[0]),
            'Era 2 (2012-2018)': int(dtm_era2.shape[0]),
            'Era 3 (2019-2025)': int(dtm_era3.shape[0])
        },
        'topic_mappings': {
            'era1_to_era2': {int(k): int(v) for k, v in mapping_1_to_2.items()},
            'era1_to_era3': {int(k): int(v) for k, v in mapping_1_to_3.items()}
        },
        'macro_topic_prevalences': [
            {
                'track_id': t_ref,
                'prevalence_era1': float(mean_weights1[t_ref]),
                'prevalence_era2': float(mean_weights2[t_ref]),
                'prevalence_era3': float(mean_weights3[t_ref])
            }
            for t_ref in range(N_TOPICS)
        ],
        'topic_tracks': json_topic_tracks
    }
    
    json_path = "results/aligned_lda_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"    [+] Saved quantitative results to: '{json_path}'")
    
    # 5. Visualizations
    print("[*] Generating comparative analytical plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Macro-Level Aligned Topic Prevalence Drift
    prevalence_data = []
    for t_ref in range(N_TOPICS):
        prevalence_data.append({
            'Topic Track': f"Track {t_ref}",
            'Mean Prevalence': float(mean_weights1[t_ref]),
            'Era': 'Era 1 (2000-2011)'
        })
        prevalence_data.append({
            'Topic Track': f"Track {t_ref}",
            'Mean Prevalence': float(mean_weights2[t_ref]),
            'Era': 'Era 2 (2012-2018)'
        })
        prevalence_data.append({
            'Topic Track': f"Track {t_ref}",
            'Mean Prevalence': float(mean_weights3[t_ref]),
            'Era': 'Era 3 (2019-2025)'
        })
    df_prevalence = pd.DataFrame(prevalence_data)
    
    plt.figure(figsize=(11, 6))
    sns.barplot(data=df_prevalence, x='Topic Track', y='Mean Prevalence', hue='Era', palette='muted')
    plt.title("Macro-Level Topic Prevalence Drift Across Three Tech Epochs (RQ1)", fontsize=14, weight='bold')
    plt.ylabel("Mean Topic Representation Weight in Documents", fontsize=12)
    plt.xlabel("Aligned Topic Track", fontsize=12)
    plt.legend(title="Tech Epoch")
    plt.tight_layout()
    
    prevalence_chart_path = "results/aligned_topic_prevalence.png"
    plt.savefig(prevalence_chart_path, dpi=300)
    plt.close()
    print(f"    [+] Saved macro-level topic prevalence chart to: '{prevalence_chart_path}'")
    
    # Plot 2: Detailed Term Probability Shifts for Era 3 Key Terms
    fig, axes = plt.subplots(5, 1, figsize=(12, 22))
    
    for t_ref in range(N_TOPICS):
        t_era2 = mapping_1_to_2[t_ref]
        t_era3 = mapping_1_to_3[t_ref]
        
        # We will use the top 5 words of Era 3 as the basis for comparison
        top_indices3 = comp3[t_era3].argsort()[:-6:-1]
        key_words = [feature_names[i] for i in top_indices3]
        
        # Get the weights of these specific 5 words across all three eras
        w1_vals = [float(comp1[t_ref][vectorizer.vocabulary_[w]]) for w in key_words]
        w2_vals = [float(comp2[t_era2][vectorizer.vocabulary_[w]]) for w in key_words]
        w3_vals = [float(comp3[t_era3][vectorizer.vocabulary_[w]]) for w in key_words]
        
        # Structure data for seaborn
        plot_data = []
        for word, w1, w2, w3 in zip(key_words, w1_vals, w2_vals, w3_vals):
            plot_data.append({'Word': word, 'Weight': w1, 'Era': 'Era 1 (2000-2011)'})
            plot_data.append({'Word': word, 'Weight': w2, 'Era': 'Era 2 (2012-2018)'})
            plot_data.append({'Word': word, 'Weight': w3, 'Era': 'Era 3 (2019-2025)'})
            
        df_plot = pd.DataFrame(plot_data)
        
        ax = axes[t_ref]
        sns.barplot(data=df_plot, x='Word', y='Weight', hue='Era', ax=ax, palette='muted')
        ax.set_title(f"Aligned Topic Track {t_ref} - Probability of Era 3 Key Terms Across Tech Epochs", fontsize=12, weight='bold')
        ax.set_ylabel("Word Probability Weight")
        ax.set_xlabel("")
        if t_ref == 0:
            ax.legend(title="Tech Epoch")
        else:
            ax.get_legend().remove()
            
    plt.suptitle("Linguistic Evolution & Weight Shift of Key AI Terms (2000–2025)", fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    
    chart_path = "results/aligned_topics_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"    [+] Saved dynamic weight comparative chart to: '{chart_path}'\n")

if __name__ == "__main__":
    main()

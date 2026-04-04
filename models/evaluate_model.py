import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer

# --- DYNAMIC PATH SETUP ---
# This finds the 'svm-based-action-extraction' root folder automatically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Update paths to look inside the /data and /models folders
MODEL_PATH = os.path.join(SCRIPT_DIR, "svm_model.pkl") 
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

DATASETS = {
    "Formal English (10k)": os.path.join(DATA_FOLDER, "action_items_dataset_10k.csv"),
    "Mixed Taglish (20k)": os.path.join(DATA_FOLDER, "action_items_dataset_20k_taglish.csv"),
    "Tagalog Set": os.path.join(DATA_FOLDER, "tagalog_action_items.csv"),
    "Taglish Set": os.path.join(DATA_FOLDER, "taglish_action_items.csv"),
    "Expanded English": os.path.join(DATA_FOLDER, "english_expanded_dataset.csv"),
    "Comprehensive (12k)": os.path.join(DATA_FOLDER, "comprehensive_thesis_dataset_12k.csv"),
    "Meeting Specific (15k)": os.path.join(DATA_FOLDER, "meeting_specific_dataset_15k.csv"),
    "Expanded Contexts (20k)": os.path.join(DATA_FOLDER, "expanded_meeting_contexts_20k.csv"),
    "Massive Diverse (50k)": os.path.join(DATA_FOLDER, "massive_diverse_dataset_50000.csv"),
    "Ultimate Diversity (50k)": os.path.join(DATA_FOLDER, "ultimate_diversity_dataset_50k.csv"),
    "AMI Corpus Dataset": os.path.join(DATA_FOLDER, "ami_dialogueActs_dataset.csv"),
    "Multilingual Resistant (Refined)": os.path.join(DATA_FOLDER, "ami_multilingual_balanced.csv")
}

# UPDATED: Matches the high-capacity Main.py (2**16 = 65536)
vectorizer = HashingVectorizer(
    n_features=2**16, 
    ngram_range=(1, 3), 
    alternate_sign=False
)

def load_and_normalize(file_path):
    """Handles the different formats of your 10k and 20k files."""
    df = pd.read_csv(file_path)
    
    # 1. Normalize Column Names (10k uses 'text', 20k uses 'sentence')
    if 'text' in df.columns:
        df.rename(columns={'text': 'sentence'}, inplace=True)
    
    # 2. Normalize Labels (10k uses strings, 20k uses 0/1)
    label_map = {'information_item': 0, 'action_item': 1}
    df['label'] = df['label'].replace(label_map)
    
    # Ensure labels are integers (0 or 1)
    df['label'] = df['label'].astype(int)
    
    return df

def run_evaluation():
    # Helpful debug info for your thesis log
    print(f"[System] Searching for model at: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found. Check if svm_model.pkl is in the 'models' folder.")
        return

    # Load your trained .pkl brain
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)

    print("\n" + "="*60)
    print("      AI PERFORMANCE SCORECARD (PATH-CORRECTED)      ")
    print("="*60)

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"⚠️  Skipping {name}: File not found at {path}")
            continue

        # Load and clean the data
        df = load_and_normalize(path)
        
        # Transform text to features
        X = vectorizer.transform(df['sentence'])
        y_true = df['label']

        # Get Predictions
        y_pred = clf.predict(X)

        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        
        print(f"\n📊 RESULTS FOR: {name}")
        print(f"Location: {path}")
        print(f"Total Samples: {len(df)}")
        print(f"Overall Accuracy: {acc * 100:.2f}%")
        print("-" * 30)
        print(classification_report(y_true, y_pred, target_names=['Information', 'Action Item']))
        print("="*60)

if __name__ == "__main__":
    run_evaluation()
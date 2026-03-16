import os
import pandas as pd
import spacy
import torch
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import queue
import time
import pickle  # For saving/loading the model
from faster_whisper import WhisperModel
from sklearn.linear_model import SGDClassifier  # Supports incremental learning
from sklearn.feature_extraction.text import HashingVectorizer # Lightweight vectorizer
from transformers import BartForConditionalGeneration, BartTokenizer

# --- INITIALIZATION ---
nlp = spacy.load("en_core_web_sm")
audio_queue = queue.Queue()
is_recording = True
fs = 16000
MODEL_PATH = "svm_model.pkl"

# We use HashingVectorizer because it doesn't need to 'fit' on a specific vocabulary
vectorizer = HashingVectorizer(n_features=2**10)

# --- PHASE 3: INCREMENTAL SVM LOAD/INIT ---
def load_or_init_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            print("[System] Loading existing self-trained model...")
            return pickle.load(f)
    else:
        print("[System] Initializing new model...")
        model = SGDClassifier(loss='hinge')
        # Initial "seed" data to establish classes [0, 1]
        X_initial = vectorizer.transform(["info", "please do this"])
        y_initial = [0, 1]
        model.partial_fit(X_initial, y_initial, classes=[0, 1])
        return model

clf = load_or_init_model()

# --- PHASE 4: ABSTRACTIVE SUMMARIZATION CLASS ---
class AbstractiveSummarizer:
    def __init__(self):
        print("Loading Phase 4: BART Summarization Model...")
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

    def generate_summary(self, text, action_items):
        input_text = f"Meeting: {text}"
        if action_items:
            # We use a clear label so BART knows these are the important parts.
            input_text += " Tasks: " + " . ".join(action_items)

        inputs = self.tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
        
        # 2. Use 'forced_bos_token_id' (This addresses the warning you saw earlier)
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=4, 
            min_length=10, 
            max_length=60, 
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3, # Prevents copying the 'instruction' if used
            early_stopping=True,
            forced_bos_token_id=0 #
        )
        
        decoded_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # 3. POST-PROCESSING: If the model STILL copies the prompt, we strip it.
        # This ensures your GUI and Terminal stay clean.
        if "Summarize the" in decoded_summary:
            decoded_summary = decoded_summary.split("list.")[-1].strip()
            
        return decoded_summary

# --- PHASE 3: FEATURE ENGINEERING ---
def get_features(sentence):
    """Returns the vectorized representation of the text."""
    return vectorizer.transform([sentence])

# --- REAL-TIME THREAD ---
def real_time_transcription():
    model = WhisperModel("base", device="cpu", compute_type="int8")
    while is_recording or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1)
            segments, _ = model.transcribe(audio_chunk, language="tl", beam_size=5)
            for segment in segments:
                if segment.text.strip():
                    print(f">>> LIVE: {segment.text.strip()}")
        except queue.Empty:
            continue

def train_from_csv(file_paths, model):
    """Reads CSV datasets and trains the SVM model with robust column detection."""
    label_map = {"action_item": 1, "information_item": 0}
    
    for path in file_paths:
        if os.path.exists(path):
            print(f"[System] Training on {path}...")
            df = pd.read_csv(path)
            
            # --- ROBUST COLUMN CLEANING ---
            # Convert all column names to lowercase and remove spaces
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Check if required columns exist after cleaning
            if 'text' not in df.columns or 'label' not in df.columns:
                print(f"[Error] {path} is missing 'text' or 'label' columns.")
                print(f"Actual columns found: {df.columns.tolist()}")
                continue # Skip this file and move to the next
            
            # Clean data: Remove empty rows
            df = df.dropna(subset=['text', 'label'])
            texts = df['text'].astype(str).tolist() # Ensure everything is a string
            labels = [label_map.get(l.strip().lower(), 0) for l in df['label'].tolist()]
            
            # Vectorize and Train
            X = vectorizer.transform(texts)
            model.partial_fit(X, labels, classes=[0, 1])
            print(f"  - Successfully processed {len(texts)} rows.")
        else:
            print(f"[Error] File not found: {path}")
            
    # Save the updated model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"[System] Training complete. Model saved to {MODEL_PATH}.")
    return model

# --- THE MAIN PIPELINE ---
def main():
    global is_recording, clf
    
    # --- MODE SELECTION ---
    print("\n" + "="*60 + "\n SYSTEM READY: SELECT MODE \n" + "="*60)
    print(" (1) Live Meeting (Use Microphone)")
    print(" (2) Process File (Use .mp4/.mp3)")
    print(" (3) Train AI Model (Use .csv datasets)")
    mode = input(" >> Enter 1, 2, or 3: ").strip()

    if mode == "3":
        datasets = ["action_items_dataset_20k_taglish.csv"]
        clf = train_from_csv(datasets, clf)
        return  # Exit early so we don't load BART or Whisper for training

    # --- Only load these for Transcription Modes (1 & 2) ---
    summarizer = AbstractiveSummarizer()
    audio_data = []
    
    if mode == "1":
        is_recording = True
        filename = "live_meeting_output.wav"
        
        def callback(indata, frames, time, status):
            audio_data.append(indata.copy())
            if len(audio_data) % 150 == 0: 
                chunk = np.concatenate(audio_data[-150:])
                audio_queue.put(chunk.flatten())

        threading.Thread(target=real_time_transcription, daemon=True).start()

        print("\n" + "█"*60 + "\n PHASE 1: LIVE RECORDING \n" + "█"*60)
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            input("\n[RECORDING] Speak into your mic. Press [ENTER] to stop...\n")

        is_recording = False
        full_audio = np.concatenate(audio_data)
        wav.write(filename, fs, full_audio)
        
    elif mode == "2":
        filename = "AI Voice Generator with Emotional Text to Speech - Google Chrome 2026-03-06 21-49-09.mp4"
        if not os.path.exists(filename):
            print(f"[Error]: {filename} not found.")
            return

   # --- PHASE 1: TRANSCRIPTION ---
    print(f"\n" + "█"*60 + "\n PHASE 1: TRANSCRIPTION (FILE MODE) \n" + "█"*60)

    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, info = model.transcribe(filename, language="tl") #

    # Convert segments to list immediately to ensure they are captured
    segments = list(segments) 

    if not segments:
        print(f"[Warning]: No speech detected in {filename}. Check if the file has audio.")
        return

    raw_text = " ".join([s.text for s in segments]) #
    print(f"\n[PHASE 1 RESULT]:\n{raw_text}")
    
    # --- PHASE 2: SEGMENTATION ---
    print(f"\n" + "█"*60 + "\n PHASE 2: TOPIC SEGMENTATION \n" + "█"*60)
    # Splitting logic to group sentences into "chunks"
    sentences = [s.strip() for s in raw_text.replace('so ', '. ').split('.') if len(s) > 5]
    topic_segments = [sentences[i:i + 5] for i in range(0, len(sentences), 5)]

    # PRINT Phase 2 results
    for i, segment in enumerate(topic_segments):
        print(f"\n[SEGMENT {i+1}]:")
        print(" ".join(segment))

    # --- PHASE 3 & SELF-TRAINING DATA COLLECTION ---
    print(f"\n" + "█"*60 + "\n PHASE 3: ACTION ITEM EXTRACTION & SELF-TRAIN \n" + "█"*60)
    all_chunks_data = []
    correction_data = []

    for chunk in topic_segments:
        detected_actions = []
        for sent in chunk:
            feat = get_features(sent)
            prediction = clf.predict(feat)[0]
            
            # Print for user review later
            status = "[!] Action" if prediction == 1 else "[ ] Info"
            print(f"  {status}: {sent}")
            
            # Save for the "Review Session"
            correction_data.append((sent, prediction))
            
            if prediction == 1:
                detected_actions.append(sent)
        
        all_chunks_data.append({"chunk_text": " ".join(chunk), "actions": detected_actions})

    # --- PHASE 4: SUMMARY ---
    print(f"\n" + "█"*60 + "\n PHASE 4: SUMMARIZATION (BART) \n" + "█"*60)
    for i, data in enumerate(all_chunks_data):
        summary = summarizer.generate_summary(data["chunk_text"], data["actions"])
        data["summary"] = summary 
        
        # PRINT Phase 4 results
        print(f"\n--- Summary for Segment {i+1} ---")
        print(f"INPUT: {data['chunk_text'][:100]}...") # Show a snippet of input
        print(f"OUTPUT: {summary}")

    # --- THE SELF-TRAINING FEEDBACK LOOP ---
    print("\n" + "="*60)
    ask_review = input("Do you want to review this session to improve the AI? (y/n): ").lower()
    
    if ask_review == 'y':
        print(f"\n" + "█"*60 + "\n SELF-TRAINING REVIEW MODE \n" + "█"*60)
        updated = False
        for sent, pred in correction_data:
            user_input = input(f"AI marked as {pred}: '{sent}' -> Correct? (y/0/1): ").lower()
            
            if user_input != 'y':
                correct_label = int(user_input)
                clf.partial_fit(get_features(sent), [correct_label])
                updated = True
                print(f"  [Learned] '{sent}' is now recognized as {correct_label}")

        if updated:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(clf, f)
            print("[System] Model updated and saved.")
    else:
        print("[System] Review skipped. Model remains unchanged.")

if __name__ == "__main__":
    main()
import os
import spacy
import torch
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import queue
import time
from faster_whisper import WhisperModel
from sklearn.svm import SVC
from transformers import BartForConditionalGeneration, BartTokenizer

# --- INITIALIZATION ---
nlp = spacy.load("en_core_web_sm")
audio_queue = queue.Queue()
is_recording = True
fs = 16000  # Standard for Whisper

# --- PHASE 4: ABSTRACTIVE SUMMARIZATION CLASS ---
class AbstractiveSummarizer:
    def __init__(self):
        print("Loading Phase 4: BART Summarization Model (this may take a moment)...")
        # 'facebook/bart-large-cnn' is the industry standard for abstractive summaries
        # For a lighter version, use 'sshleifer/distilbart-cnn-12-6'
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

    def generate_summary(self, text, action_items):
        # Step 1: Combine context and detected action items for 'Coherence' (Dual-stage)
        context_with_actions = text
        if action_items:
            action_context = " Key tasks identified: " + " ".join(action_items)
            context_with_actions += action_context

        # Step 2: Tokenize and Generate
        inputs = self.tokenizer([context_with_actions], max_length=1024, return_tensors="pt", truncation=True)
        
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=4, 
            min_length=10,  # Lowered to allow condensation
            max_length=60,  # Tightened to force BART to summarize, not copy
            length_penalty=2.0,
            early_stopping=True
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- PHASE 3: SVM TRAINING (Mock Data for Thesis) ---
clf = SVC(kernel='linear')
train_X = [[1,0,0,0], [0,0,0,0], [0,1,0,1], [0,0,0,0]] 
train_y = [1, 0, 1, 0] # 1 = Action Item, 0 = Information
clf.fit(train_X, train_y)

# --- PHASE 3: FEATURE ENGINEERING ---
def get_features(sentence):
    text = sentence.lower()
    doc = nlp(text)
    
    # Feature 1: Presence of task keywords (Taglish + Academic English)
    action_keywords = ['paki', 'sige', 'review', 'read', 'prepare', 'submit', 'expect', 'priority', 'must']
    has_action_word = 1 if any(word in text for word in action_keywords) else 0
    
    # Feature 2: Time/Deadline markers
    has_deadline = 1 if any(word in text for word in ['due', 'before', 'monday', 'thursday', '2 p.m', 'midnight']) else 0
    
    # Feature 3: Starts with Verb (Imperative)
    starts_with_verb = 1 if (len(doc) > 0 and doc[0].pos_ == "VERB") else 0
    
    return [has_action_word, has_deadline, starts_with_verb, 0]

# --- RE-TRAIN SVM WITH BETTER DATA ---
# Let's give it a few more examples so it understands the academic context
train_X = [
    [1, 1, 1, 0], # "Please review the slides before Thursday" (Action)
    [1, 0, 1, 0], # "Read the article" (Action)
    [0, 0, 0, 0], # "Alright everyone" (Info)
    [0, 1, 0, 0], # "The portal is live" (Info)
    [1, 1, 0, 0], # "Your problem sets are due Monday" (Action)
]
train_y = [1, 1, 0, 0, 1] 
clf.fit(train_X, train_y)

# --- REAL-TIME THREAD (The "Live" Experience) ---
def real_time_transcription():
    """Uses Faster-Whisper to transcribe 3-second chunks while recording."""
    # INT8 makes it 4x faster on a CPU
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    while is_recording or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=1)
            # 'tl' forces Taglish recognition
            segments, _ = model.transcribe(audio_chunk, language="tl", beam_size=5)
            for segment in segments:
                if segment.text.strip():
                    print(f">>> LIVE: {segment.text.strip()}")
        except queue.Empty:
            continue

# --- THE MAIN PIPELINE ---
def main():
    global is_recording
    filename = "meeting_recording.wav"
    # MP4 for testing
    # video_path = r"C:\Users\John Keith Mercado\Downloads\20260221__Alright_e.mp3"
    summarizer = AbstractiveSummarizer() # Initialize BART
    audio_data = []
    # redirect to output folder for integrations of done output files
    # separate with date and time for uniqueness
    # can be as well a training dataset for the future SVM improvements

    print("\n" + "="*60)
    print(" SYSTEM READY: LOCAL TAGLISH TRANSCRIPTION ")
    print("="*60)

    # Audio Callback to capture mic data
    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())
        # Every 3 seconds, send chunk to the Live Thread
        if len(audio_data) % 150 == 0: 
            chunk = np.concatenate(audio_data[-150:])
            audio_queue.put(chunk.flatten())

    # Start the Live Thread
    threading.Thread(target=real_time_transcription, daemon=True).start()

    # Start Recording
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        input("\n[RECORDING STARTED] Speak into your mic. Press [ENTER] to stop...\n")

    is_recording = False
    print("\nProcessing final results... please wait.")

    # Save Full Recording
    full_audio = np.concatenate(audio_data)
    wav.write(filename, fs, full_audio)

    # --- PHASE 1: BATCH TRANSCRIPTION ---
    print(f"\n" + "█"*60)
    print(f" PHASE 1: FULL TRANSCRIPTION (FASTER-WHISPER) ")
    print(f"█"*60)
    
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(filename, language="tl")
    
    # Reconstruct the full text result from segments
    raw_text = " ".join([s.text for s in segments])
    print("\n[FULL RAW TEXT]:")
    print(raw_text)

    # --- PHASE 2: TOPIC SEGMENTATION ---
    print(f"\n" + "█"*60 + "\n PHASE 2: TOPIC SEGMENTATION \n" + "█"*60)
    sentences = [s.strip() for s in raw_text.replace('so ', '. ').split('.') if len(s) > 5]
    topic_segments = [sentences[i:i + 5] for i in range(0, len(sentences), 5)]
    for i, segment in enumerate(topic_segments):
        print(f"\n[TOPIC CHUNK {i+1}]")
        for line in segment: print(f"  • {line}")

# --- PHASE 3: ACTION ITEM EXTRACTION ---
    print(f"\n" + "█"*60 + "\n PHASE 3: ACTION ITEM EXTRACTION (SVM) \n" + "█"*60)
    all_chunks_data = []

    for i, chunk in enumerate(topic_segments):
        print(f"\n[Scanning Segment {i+1}]")
        detected_actions = []
        
        for sent in chunk:
            feat = [get_features(sent)]
            prediction = clf.predict(feat)[0]
            
            if prediction == 1:
                label = "[!] Action item"
                detected_actions.append(sent)
            else:
                label = "[ ] Info"
            
            print(f"  {label}: {sent}")
        
        all_chunks_data.append({"chunk_text": " ".join(chunk), "actions": detected_actions})

    # --- PHASE 4: ABSTRACTIVE SUMMARIZATION ---
    print(f"\n" + "█"*60 + "\n PHASE 4: ABSTRACTIVE SUMMARIZATION (BART) \n" + "█"*60)
    for i, data in enumerate(all_chunks_data):
        print(f"\n[Generating Summary for Segment {i+1}...]")
        summary = summarizer.generate_summary(data["chunk_text"], data["actions"])
        data["summary"] = summary # Save for final report
        print(f"RESULTING SUMMARY: {summary}")

    # --- FINAL COMBINED OUTPUT ---
    print(f"\n" + "="*60 + "\n FINAL GENERATED MINUTES OF THE MEETING \n" + "="*60)
    for i, entry in enumerate(all_chunks_data):
        print(f"\nTOPIC SECTION {i+1}")
        print(f"Key Points: {entry['summary']}")
        if entry['actions']:
            print("Action Items:")
            for act in entry['actions']: print(f"  • {act}")

if __name__ == "__main__":
    main()
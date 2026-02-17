import os
import spacy
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import queue
import time
from faster_whisper import WhisperModel
from sklearn.svm import SVC

# --- INITIALIZATION ---
nlp = spacy.load("en_core_web_sm")
audio_queue = queue.Queue()
is_recording = True
fs = 16000  # Standard for Whisper

# --- PHASE 3: SVM TRAINING (Mock Data for Thesis) ---
clf = SVC(kernel='linear')
train_X = [[1,0,0,0], [0,0,0,0], [0,1,0,1], [0,0,0,0]] 
train_y = [1, 0, 1, 0] # 1 = Action Item, 0 = Information
clf.fit(train_X, train_y)

# --- PHASE 3: FEATURE ENGINEERING ---
def get_features(sentence):
    text = sentence.lower()
    doc = nlp(text)
    first_token_pos = doc[0].pos_ if len(doc) > 0 else ""
    return [
        1 if 'paki' in text else 0,              # Taglish Politeness
        1 if 'sige' in text else 0,              # Agreement/Volunteer
        1 if first_token_pos == "VERB" else 0,   # Imperative Command
        1 if any(m in text for m in ['need', 'must', 'will']) else 0 # Modals
    ]

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
    audio_data = []

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
    print(f"\n" + "█"*60)
    print(f" PHASE 2: TOPIC SEGMENTATION RESULTS ")
    print(f"█"*60)
    
    # Split text into clean sentences
    sentences = [s.strip() for s in raw_text.replace('so ', '. ').split('.') if len(s) > 5]
    
    # Segmenting into Topic Chunks (Every 5 sentences)
    topic_segments = [sentences[i:i + 5] for i in range(0, len(sentences), 5)]
    
    for i, segment in enumerate(topic_segments):
        print(f"\n[TOPIC CHUNK {i+1}]")
        for line in segment:
            print(f"  • {line}")

    # --- PHASE 3: ACTION ITEM DETECTION ---
    print(f"\n" + "█"*60)
    print(f" PHASE 3: ACTION ITEM DETECTION (SVM) ")
    print(f"█"*60)
    
    action_items = []
    
    for chunk in topic_segments:
        for sent in chunk:
            feat = [get_features(sent)]
            prediction = clf.predict(feat)[0]
            
            if prediction == 1:
                action_items.append(sent)
                print(f" [!] ACTION ITEM: {sent}")
            else:
                print(f" [ ] info: {sent[:60]}...")

    # --- FINAL REPORT ---
    print(f"\n" + "="*60)
    print(" FINAL ACTION ITEM SUMMARY ")
    print("-" * 60)
    if action_items:
        for i, item in enumerate(action_items):
            print(f"{i+1}. {item}")
    else:
        print("No action items detected.")
    print("="*60)

if __name__ == "__main__":
    main()
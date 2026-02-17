import whisper

def transcribe_free(file_path):
    print("--- PHASE 1: LOADING LOCAL WHISPER MODEL (FREE) ---")
    # 'base' is fast. 'small' or 'medium' is better for Taglish.
    model = whisper.load_model("base") 
    
    print("Transcribing... this uses your laptop's CPU/GPU.")
    # The 'r' fixed your path error!
    result = model.transcribe(file_path, fp16=False, language="tl")
    
    return result["text"]

# --- THE EXECUTION PART ---
# This part actually runs the function and prints the result
if __name__ == "__main__":
    my_file = r"C:\Users\John Keith Mercado\Downloads\Sample_Video_To_Practice_Transcribing_48kbps.mp4"
    
    transcript = transcribe_free(my_file)
    
    print("\n--- TRANSCRIPTION RESULT ---")
    print(transcript)
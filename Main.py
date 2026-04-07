"""
ThesisModel Main Entry Point
Orchestrates the multi-phase meeting transcript processing pipeline.

Phases:
  1. Transcription: Convert audio/video to text (Groq or Local Whisper)
  2. Segmentation: Group sentences into topic-based segments
  3. Classification: Identify action items vs. information items (SVM)
  4. Summarization: Generate abstractive summaries (BART or Groq)
"""

import os
import sys
import traceback
import threading

# Import core modules
from core.transcriber import Transcriber
from core.segmenter import Segmenter
from core.classifier import ActionItemClassifier
from core.summarizer import Summarizer

# Import utilities
from utils.audio_utils import AudioRecorder
from utils.trainer import ModelTrainer


def print_phase_header(phase_num, phase_name):
    """Pretty-print a phase header."""
    header = f"\n{'█' * 60}\n PHASE {phase_num}: {phase_name} \n{'█' * 60}"
    print(header)


def select_mode():
    """Get user's operating mode."""
    print("\n" + "=" * 60 + "\n SYSTEM READY: SELECT MODE \n" + "=" * 60)
    print(" (1) Live Meeting (Use Microphone)")
    print(" (2) Process File (Use .mp4/.mp3/.wav)")
    print(" (3) Train AI Model (Use .csv datasets)")
    mode = input(" >> Enter 1, 2, or 3: ").strip()
    return mode


def select_transcription_engine():
    """Get user's transcription engine choice."""
    print("\n Choose Transcription Engine")
    print(" (1) Local Faster-Whisper (offline)")
    print(" (2) Groq whisper-large-v3-turbo (API)")
    engine_choice = input(" >> Enter 1 or 2: ").strip()
    return "groq" if engine_choice == "2" else "local"


def select_summarization_engine():
    """Get user's summarization engine choice."""
    print("\n Choose Summarization Engine")
    print(" (1) Groq Llama (fast API)")
    print(" (2) Local BART (slow on CPU)")
    summary_choice = input(" >> Enter 1 or 2: ").strip()
    return "groq" if summary_choice == "1" else "local"


def mode_train(classifier, trainer):
    """
    Mode 3: Train the AI model from CSV datasets.

    Args:
        classifier: ActionItemClassifier instance
        trainer: ModelTrainer instance
    """
    print_phase_header(1, "MODEL TRAINING")
    datasets = trainer.get_training_datasets()
    trainer.train_from_csv(datasets)


def mode_live_meeting(
    transcriber, segmenter, classifier, summarizer, transcription_engine
):
    """
    Mode 1: Live meeting transcription and processing.

    Args:
        transcriber: Transcriber instance
        segmenter: Segmenter instance
        classifier: ActionItemClassifier instance
        summarizer: Summarizer instance
        transcription_engine: "local" or "groq"
    """
    print_phase_header(1, "LIVE RECORDING")

    # Record audio
    recorder = AudioRecorder(sample_rate=16000)
    filename = "live_meeting_output.wav"

    if transcription_engine == "groq":
        print(
            "[System] Groq mode in Live Meeting transcribes after you press ENTER to stop recording."
        )

    # Start real-time transcription for local engine
    if transcription_engine == "local":
        # TODO: Implement real-time transcription thread for local whisper
        pass

    # Record
    recorder.start_recording()
    recorder.save_recorded_audio(filename)

    # Process the recorded file
    process_file(
        filename,
        transcriber,
        segmenter,
        classifier,
        summarizer,
        transcription_engine,
    )


def mode_file_processing(
    transcriber, segmenter, classifier, summarizer, transcription_engine
):
    """
    Mode 2: Process a pre-recorded file.

    Args:
        transcriber: Transcriber instance
        segmenter: Segmenter instance
        classifier: ActionItemClassifier instance
        summarizer: Summarizer instance
        transcription_engine: "local" or "groq"
    """
    filename = input(" >> Enter file path (.mp4/.mp3/.wav): ").strip().strip('"')
    if not filename:
        print("[Error]: No file path provided.")
        return
    if not os.path.exists(filename):
        print(f"[Error]: {filename} not found.")
        return

    process_file(
        filename,
        transcriber,
        segmenter,
        classifier,
        summarizer,
        transcription_engine,
    )


def process_file(
    filename, transcriber, segmenter, classifier, summarizer, transcription_engine
):
    """
    Core processing pipeline for a file.

    Args:
        filename: Path to audio/video file
        transcriber: Transcriber instance
        segmenter: Segmenter instance
        classifier: ActionItemClassifier instance
        summarizer: Summarizer instance
        transcription_engine: "local" or "groq"
    """
    # --- PHASE 1: TRANSCRIPTION ---
    print_phase_header(1, "TRANSCRIPTION")

    try:
        raw_text = transcriber.transcribe_file(
            filename, engine=transcription_engine, language="tl"
        )
    except Exception as e:
        print(f"[Error]: Transcription failed -> {e}")
        traceback.print_exc()

        if transcription_engine == "groq":
            use_fallback = (
                input("[System] Do you want to fallback to local whisper? (y/n): ")
                .strip()
                .lower()
            )
            if use_fallback == "y":
                try:
                    raw_text = transcriber.transcribe_file(
                        filename, engine="local", language="tl"
                    )
                except Exception as local_err:
                    print(f"[Error]: Local fallback also failed -> {local_err}")
                    return
            else:
                return
        else:
            return

    if not raw_text.strip():
        print(f"[Warning]: No speech detected in {filename}. Check if the file has audio.")
        return

    print(f"\n[PHASE 1 RESULT]:\n{raw_text}\n")

    # --- PHASE 2: SEGMENTATION ---
    print_phase_header(2, "TOPIC SEGMENTATION")
    topic_segments = segmenter.segment_text(raw_text)
    segmenter.print_segments(topic_segments)

    # --- PHASE 3: ACTION ITEM EXTRACTION ---
    print_phase_header(3, "ACTION ITEM EXTRACTION & SELF-TRAIN")
    all_chunks_data = []
    correction_data = []

    for chunk in topic_segments:
        result = classifier.classify_segment(chunk)
        all_chunks_data.append(
            {
                "chunk_text": " ".join(chunk),
                "actions": result["detected_actions"],
                "classified_sentences": result["classified_sentences"],
            }
        )

        # Store for review session
        for item in result["classified_sentences"]:
            correction_data.append((item["sentence"], item["label"]))

    # --- PHASE 4: SUMMARIZATION ---
    print_phase_header(4, "SUMMARIZATION")
    for i, data in enumerate(all_chunks_data, 1):
        summary = summarizer.generate_summary(data["chunk_text"], data["actions"])
        data["summary"] = summary

        # Print Phase 4 results
        print(f"\n--- Summary for Segment {i} ---")
        print(f"INPUT: {data['chunk_text'][:100]}...")
        print(f"OUTPUT: {summary}")

    # --- SELF-TRAINING REVIEW MODE ---
    print("\n" + "=" * 60)
    ask_review = input(
        "Do you want to review this session to improve the AI? (y/n): "
    ).lower()

    if ask_review == "y":
        print("\n" + "█" * 60 + "\n SELF-TRAINING REVIEW MODE \n" + "█" * 60)
        trainer = ModelTrainer(classifier)
        new_corrections = trainer.collect_user_corrections(correction_data)

        if new_corrections:
            trainer.save_corrections_to_csv(new_corrections)
    else:
        print("[System] Review skipped.")


def main():
    """Main entry point."""
    try:
        # Initialize core components
        transcriber = Transcriber()
        segmenter = Segmenter(chunk_size=5)
        classifier = ActionItemClassifier()
        trainer = ModelTrainer(classifier)

        # Get user's mode
        mode = select_mode()

        if mode == "3":
            # Training mode
            mode_train(classifier, trainer)
            return

        if mode == "1":
            # Live meeting mode
            transcription_engine = select_transcription_engine()
            summarization_engine = select_summarization_engine()
            summarizer = Summarizer(engine=summarization_engine)

            mode_live_meeting(
                transcriber, segmenter, classifier, summarizer, transcription_engine
            )

        elif mode == "2":
            # File processing mode
            transcription_engine = select_transcription_engine()
            summarization_engine = select_summarization_engine()
            summarizer = Summarizer(engine=summarization_engine)

            mode_file_processing(
                transcriber, segmenter, classifier, summarizer, transcription_engine
            )

        else:
            print("[Error] Invalid mode. Please enter 1, 2, or 3.")

    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
# ThesisModel Refactoring Guide

## Overview
The monolithic `Main.py` has been refactored into a professional, modular architecture with clear separation of concerns. Each processing phase is now an independent module that can be tested and maintained separately.

---

## New Project Structure

```
ThesisModel/
├── main.py                          # Entry point (orchestrator)
├── core/                            # Core processing modules
│   ├── __init__.py
│   ├── transcriber.py               # Phase 1: Transcription (Groq/Local Whisper)
│   ├── segmenter.py                 # Phase 2: Topic segmentation
│   ├── classifier.py                # Phase 3: Action item classification (SVM)
│   └── summarizer.py                # Phase 4: Abstractive summarization
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── audio_utils.py               # Audio recording, FFmpeg chunking
│   └── trainer.py                   # Model training and dataset management
├── data/                            # (existing) Training datasets
├── integrations/                    # (existing) Third-party integrations
└── [other existing files]
```

---

## Module Descriptions

### `core/transcriber.py` - Phase 1: Transcription
**Responsibility:** Convert audio/video files to text using either Groq API or Local Whisper.

**Key Classes:**
- `Transcriber`: Manages transcription engine switching and retry logic
  - `transcribe_file()`: Main transcription method
  - `transcribe_with_groq_retries()`: Groq with retry mechanism
  - Lazy-loads local Whisper model on demand

**Benefits:**
- Encapsulates all transcription logic
- Supports engine switching without UI changes
- Built-in retry and fallback mechanisms

**Usage:**
```python
from core.transcriber import Transcriber

transcriber = Transcriber()
text = transcriber.transcribe_file("meeting.mp4", engine="groq", language="tl")
```

---

### `core/segmenter.py` - Phase 2: Topic Segmentation
**Responsibility:** Break raw transcribed text into logical topic-based segments.

**Key Classes:**
- `Segmenter`: Groups sentences into configurable-size chunks
  - `segment_text()`: Core segmentation logic
  - `print_segments()`: Pretty-print segments

**Benefits:**
- Configurable chunk size (default: 5 sentences)
- Reusable segmentation logic
- Testable in isolation

**Usage:**
```python
from core.segmenter import Segmenter

segmenter = Segmenter(chunk_size=5)
segments = segmenter.segment_text(raw_text)
segmenter.print_segments(segments)
```

---

### `core/classifier.py` - Phase 3: Action Item Classification
**Responsibility:** Use SVM to classify sentences as action items or informational text.

**Key Classes:**
- `ActionItemClassifier`: Manages SVM model, vectorizer, and predictions
  - `predict()`: Classify a single sentence
  - `classify_segment()`: Classify entire segment
  - `train_on_batch()`: Incremental training
  - `apply_correction()`: Self-training from user feedback

**Features:**
- Lazy model loading from disk
- High-capacity vectorizer (2^16 features, 1-3 grams)
- Support for incremental learning
- Self-training capability

**Usage:**
```python
from core.classifier import ActionItemClassifier

classifier = ActionItemClassifier(model_path="svm_model.pkl")
label = classifier.predict("Please do this action")  # Returns 0 or 1
results = classifier.classify_segment(["sentence1", "sentence2"])
classifier.apply_correction("sentence", 1)  # Self-training
classifier.save_model()
```

---

### `core/summarizer.py` - Phase 4: Abstractive Summarization
**Responsibility:** Generate abstractive summaries using BART or Groq API.

**Key Classes:**
- `LocalSummarizer`: BART-based local summarization
- `Summarizer`: Unified interface supporting both engines
  - `generate_summary()`: Main summarization method
  - `switch_engine()`: Runtime engine switching
  - Automatic fallback mechanism

**Benefits:**
- Engine-agnostic API
- Graceful degradation when Groq unavailable
- BART model lazy-loads on first use

**Usage:**
```python
from core.summarizer import Summarizer

summarizer = Summarizer(engine="groq")
summary = summarizer.generate_summary(
    "meeting text", 
    action_items=["task1", "task2"]
)
summarizer.switch_engine("local")
```

---

### `utils/audio_utils.py` - Audio Handling
**Responsibility:** Real-time audio recording, FFmpeg chunking for large files.

**Key Classes/Functions:**
- `AudioRecorder`: Manages microphone input and audio buffering
  - `start_recording()`: Record from microphone
  - `save_recorded_audio()`: Save to WAV file
  - `get_audio_queue()`: Access to audio chunks for real-time processing

- **Utility Functions:**
  - `can_use_ffmpeg()`: Check FFmpeg availability
  - `is_video_file()`: Detect video formats
  - `transcribe_with_groq_chunked()`: Split large files for faster Groq processing

**Benefits:**
- Clean separation of audio handling
- Support for large file compression
- Real-time audio queue system

**Usage:**
```python
from utils.audio_utils import AudioRecorder, transcribe_with_groq_chunked

# Live recording
recorder = AudioRecorder(sample_rate=16000)
recorder.start_recording()
recorder.save_recorded_audio("output.wav")

# Chunked Groq transcription
text = transcribe_with_groq_chunked("large_video.mp4", language="tl")
```

---

### `utils/trainer.py` - Model Training & Dataset Management
**Responsibility:** Load CSV datasets, perform batch training, manage user corrections.

**Key Classes:**
- `ModelTrainer`: Orchestrates training workflows
  - `train_from_csv()`: Batch training from multiple CSV files
  - `collect_user_corrections()`: Interactive self-training feedback
  - `save_corrections_to_csv()`: Persist corrections for future training
  - `get_training_datasets()`: Auto-detect available datasets

**Features:**
- Robust column detection (handles various naming conventions)
- Automatic user correction CSV discovery
- Batch and incremental training support

**Usage:**
```python
from utils.trainer import ModelTrainer
from core.classifier import ActionItemClassifier

classifier = ActionItemClassifier()
trainer = ModelTrainer(classifier)

# Batch training
datasets = trainer.get_training_datasets()
trainer.train_from_csv(datasets)

# Self-training from corrections
corrections = trainer.collect_user_corrections(correction_data)
trainer.save_corrections_to_csv(corrections)
```

---

### `main.py` - Entry Point (Orchestrator)
**Responsibility:** Present UI, coordinate modules, orchestrate the complete pipeline.

**Key Functions:**
- `main()`: Initialize all components and handle mode selection
- `select_mode()`: User chooses between Live, File, or Train mode
- `select_transcription_engine()`: Groq or Local Whisper
- `select_summarization_engine()`: Groq or Local BART
- `process_file()`: Core 4-phase pipeline
- `mode_train()`, `mode_live_meeting()`, `mode_file_processing()`: Mode-specific handlers

**Benefits:**
- Clean, readable orchestration
- Each phase clearly separated
- Easy to add new modes or modify flows

---

## Usage Examples

### Mode 1: Live Meeting
```bash
python main.py
# Select: 1
# Select transcription engine: 1 (local) or 2 (groq)
# Select summarization engine: 1 (groq) or 2 (local/BART)
# [Records from microphone and processes]
```

### Mode 2: Process Pre-recorded File
```bash
python main.py
# Select: 2
# Enter file path: meeting.mp4
# [Processes through all 4 phases]
```

### Mode 3: Train from Datasets
```bash
python main.py
# Select: 3
# [Trains on all available datasets including user corrections]
```

---

## Testing Individual Modules

Each module can now be tested independently:

```python
# Test transcription
from core.transcriber import Transcriber
transcriber = Transcriber()
text = transcriber.transcribe_file("test.mp3", engine="local")
assert len(text) > 0

# Test segmentation
from core.segmenter import Segmenter
segmenter = Segmenter()
segments = segmenter.segment_text("sentence1. sentence2. sentence3.")
assert len(segments) > 0

# Test classification
from core.classifier import ActionItemClassifier
classifier = ActionItemClassifier()
label = classifier.predict("please complete this task")
assert label in [0, 1]

# Test summarization
from core.summarizer import Summarizer
summarizer = Summarizer(engine="local")
summary = summarizer.generate_summary("meeting text", [])
assert len(summary) > 0
```

---

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **File Size** | 600+ lines (monolithic) | Modular (50-200 lines each) |
| **Testability** | Difficult (tightly coupled) | Easy (independent modules) |
| **Maintainability** | Hard to modify | Clean, focused modules |
| **Code Reuse** | Limited | High (modular components) |
| **Engine Switching** | Requires code changes | Runtime configuration |
| **Error Handling** | Mixed concerns | Encapsulated per module |
| **Documentation** | Implicit | Explicit docstrings |

---

## Migration Notes

- **Backward Compatibility**: The new `main.py` provides the same user experience
- **Model Files**: Existing `svm_model.pkl` is compatible
- **Datasets**: All existing CSV files work without changes
- **Dependencies**: No new packages required (uses existing ones)

---

## Future Enhancements

1. **API Layer**: Create Flask/FastAPI wrapper for headless operation
2. **Testing Suite**: Unit tests for each module
3. **Configuration File**: YAML/JSON config instead of interactive prompts
4. **Logging**: Structured logging to file
5. **Performance Metrics**: Track accuracy, processing time per phase
6. **Real-time Dashboard**: Web UI for live meeting monitoring
7. **Database Backend**: Store sessions and corrections in DB instead of CSV

---

## Quick Reference

| Module | Purpose | Key Class/Function |
|--------|---------|-------------------|
| `transcriber.py` | Groq/Local transcription | `Transcriber.transcribe_file()` |
| `segmenter.py` | Topic segmentation | `Segmenter.segment_text()` |
| `classifier.py` | SVM action classification | `ActionItemClassifier.predict()` |
| `summarizer.py` | BART/Groq summarization | `Summarizer.generate_summary()` |
| `audio_utils.py` | Recording & chunking | `AudioRecorder`, `transcribe_with_groq_chunked()` |
| `trainer.py` | Training & corrections | `ModelTrainer.train_from_csv()` |
| `main.py` | Orchestrator | `main()`, `process_file()` |

---

**Refactoring Date:** 2026-04-07  
**Structure Version:** 2.0 (Professional Modular Architecture)

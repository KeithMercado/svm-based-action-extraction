# Quick Start Guide - Refactored ThesisModel

## Running the Application

### Option 1: Interactive Mode (Recommended)
```bash
cd ThesisModel
python main.py
```

Then follow the prompts to select:
1. **Mode** (1=Live, 2=File, 3=Train)
2. **Transcription Engine** (1=Local, 2=Groq)
3. **Summarization Engine** (1=Groq, 2=Local)

---

## Module Import Examples

### Using Individual Modules Programmatically

```python
# Import Phase 1: Transcription
from core.transcriber import Transcriber

transcriber = Transcriber()
text = transcriber.transcribe_file(
    "meeting.mp4", 
    engine="groq",  # or "local"
    language="tl"
)
```

```python
# Import Phase 2: Segmentation
from core.segmenter import Segmenter

segmenter = Segmenter(chunk_size=5)
segments = segmenter.segment_text(transcribed_text)
for i, segment in enumerate(segments):
    print(f"Segment {i}: {' '.join(segment)}")
```

```python
# Import Phase 3: Classification
from core.classifier import ActionItemClassifier

classifier = ActionItemClassifier()
label = classifier.predict("Please complete this task")  # Returns 0 or 1

# Classify entire segment
results = classifier.classify_segment(["sentence1", "sentence2", "sentence3"])
print(results['detected_actions'])  # List of action items

# Self-training
classifier.apply_correction("sentence text", correct_label=1)
classifier.save_model()
```

```python
# Import Phase 4: Summarization
from core.summarizer import Summarizer

summarizer = Summarizer(engine="groq")  # or "local"
summary = summarizer.generate_summary(
    text="meeting transcript",
    action_items=["task1", "task2"]
)
```

```python
# Import Audio Utilities
from utils.audio_utils import AudioRecorder, transcribe_with_groq_chunked

# Record live audio
recorder = AudioRecorder(sample_rate=16000)
full_audio = recorder.start_recording()
recorder.save_recorded_audio("output.wav")

# Chunk and transcribe large files
text = transcribe_with_groq_chunked("video.mp4", language="tl")
```

```python
# Import Model Training
from utils.trainer import ModelTrainer
from core.classifier import ActionItemClassifier

classifier = ActionItemClassifier()
trainer = ModelTrainer(classifier)

# Auto-discover and train on all datasets
datasets = trainer.get_training_datasets()
trainer.train_from_csv(datasets)

# Process user corrections
corrections = trainer.collect_user_corrections(correction_data)
trainer.save_corrections_to_csv(corrections)
```

---

## Building Custom Pipelines

### Example: Custom Processing Script

```python
from core.transcriber import Transcriber
from core.segmenter import Segmenter
from core.classifier import ActionItemClassifier
from core.summarizer import Summarizer

# Initialize components
transcriber = Transcriber()
segmenter = Segmenter(chunk_size=5)
classifier = ActionItemClassifier()
summarizer = Summarizer(engine="local")

# Process file
filename = "meeting.mp4"

# Phase 1: Transcribe
print("Transcribing...")
raw_text = transcriber.transcribe_file(
    filename, 
    engine="local",  # Use local for offline processing
    language="tl"
)

# Phase 2: Segment
print("Segmenting...")
segments = segmenter.segment_text(raw_text)

# Phase 3: Classify
print("Classifying...")
all_actions = []
for segment in segments:
    result = classifier.classify_segment(segment)
    all_actions.extend(result["detected_actions"])

# Phase 4: Summarize
print("Summarizing...")
summary = summarizer.generate_summary(raw_text, action_items=all_actions)

# Output results
print(f"\nAction Items: {all_actions}")
print(f"\nSummary: {summary}")
```

---

## Error Handling Example

```python
from core.transcriber import Transcriber
import traceback

transcriber = Transcriber()

try:
    # Try Groq first
    text = transcriber.transcribe_file("file.mp4", engine="groq")
except Exception as e:
    print(f"Groq failed: {e}")
    try:
        # Fallback to local
        print("Falling back to local transcription...")
        text = transcriber.transcribe_file("file.mp4", engine="local")
    except Exception as local_error:
        print(f"Local transcription also failed: {local_error}")
        traceback.print_exc()
```

---

## Testing Individual Modules

```bash
# Test if imports work
python -c "from core.transcriber import Transcriber; print('✓ Transcriber imported')"
python -c "from core.segmenter import Segmenter; print('✓ Segmenter imported')"
python -c "from core.classifier import ActionItemClassifier; print('✓ Classifier imported')"
python -c "from core.summarizer import Summarizer; print('✓ Summarizer imported')"
python -c "from utils.audio_utils import AudioRecorder; print('✓ AudioRecorder imported')"
python -c "from utils.trainer import ModelTrainer; print('✓ ModelTrainer imported')"
```

---

## File Structure Verification

Expected structure after refactoring:

```
ThesisModel/
├── main.py                              # ← Entry point
├── REFACTORING_GUIDE.md
├── QUICK_START.md                       # ← You are here
├── core/
│   ├── __init__.py
│   ├── transcriber.py                   # Phase 1
│   ├── segmenter.py                     # Phase 2
│   ├── classifier.py                    # Phase 3
│   └── summarizer.py                    # Phase 4
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py
│   └── trainer.py
├── data/
│   └── [CSV datasets]
├── integrations/
│   └── groq/
│       ├── transcribe.py
│       └── summarize.py
└── [other existing files]
```

---

## Environment Setup

Ensure your Python environment has all dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `faster-whisper` (Phase 1 local transcription)
- `transformers` + `torch` (Phase 4 BART summarization)
- `scikit-learn` (Phase 3 SVM classifier)
- `spacy` (NLP utilities)
- `pandas` (Data handling)
- `sounddevice` + `scipy` (Audio recording)

---

## Troubleshooting

### Import Errors
If you get "ModuleNotFoundError", ensure:
1. You're running from the `ThesisModel` directory
2. `core/` and `utils/` directories exist with `__init__.py` files
3. Python path includes the project root

### Model Loading Issues
```python
# Force model reinitialization
from core.classifier import ActionItemClassifier
classifier = ActionItemClassifier(model_path="new_model.pkl")
```

### Groq Integration
Ensure `integrations/groq/transcribe.py` and `integrations/groq/summarize.py` exist and are properly configured with your API keys.

---

## Next Steps

1. **Read** [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) for detailed module documentation
2. **Run** `python main.py` to test the refactored system
3. **Explore** individual modules programmatically
4. **Extend** by creating custom modules that inherit from base classes

---

**Happy refactoring!** 🚀

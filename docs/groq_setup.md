# Groq Setup Guide

## 1) Install dependencies

Run:

```bash
pip install groq python-dotenv
```

If you want to pin them, add to requirements.txt:

- groq
- python-dotenv

## 2) Configure environment variables

1. Copy `.env.example` to `.env`
2. Put your key:

```env
GROQ_API_KEY=your_real_key_here
```

Your `.gitignore` already ignores `.env`, which is correct.

## 3) Where to store keys

- Put real keys only in `.env` at project root.
- Do not hardcode keys in Python files.
- Do not paste keys in notes or commits.

## 4) Use Whisper Large V3 Turbo

This project includes a helper script at:

- `integrations/groq/transcribe.py`

Run it:

```bash
python integrations/groq/transcribe.py "path/to/audio.mp3" tl
```

Model used in code:

- `whisper-large-v3-turbo`

## 5) Integrate into your app

You can import this function anywhere:

```python
from integrations.groq.transcribe import transcribe_with_groq

text = transcribe_with_groq("output/videos/sample.wav", language="tl")
```

Then pass `text` to your existing action-item pipeline.

## 6) Is it free?

Groq pricing can change. Usually there is a free developer allowance/rate-limited usage, but it is not an unlimited free service.

Check your live status here before production use:

- Groq Console usage dashboard
- Groq pricing page

Always monitor usage limits so your thesis demo does not fail unexpectedly.

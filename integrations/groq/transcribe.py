import os
from pathlib import Path
from typing import Optional


MODEL_NAME = "whisper-large-v3-turbo"


def transcribe_with_groq(
    audio_file_path: str,
    language: str = "tl",
    initial_prompt: Optional[str] = None,
) -> str:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: python-dotenv. Run: pip install python-dotenv"
        ) from e

    try:
        from groq import Groq
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: groq. Run: pip install groq"
        ) from e

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

    file_path = Path(audio_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    client = Groq(api_key=api_key)

    request_kwargs = {
        "model": MODEL_NAME,
        "language": language,
        "response_format": "verbose_json",
        "temperature": 0,
    }
    if initial_prompt:
        request_kwargs["prompt"] = initial_prompt

    with file_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            **request_kwargs,
        )

    text = getattr(result, "text", "")
    if not text:
        raise RuntimeError("No transcript text returned by Groq API.")

    return text


if __name__ == "__main__":
    # Quick CLI usage:
    # python integrations/groq/transcribe.py path/to/file.mp3 tl
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python integrations/groq/transcribe.py <audio_path> [language]")

    audio_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "tl"

    transcript = transcribe_with_groq(audio_path, language=lang)
    print("\n--- GROQ TRANSCRIPT ---\n")
    print(transcript)

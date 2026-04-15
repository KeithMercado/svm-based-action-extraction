import os
import re
from getpass import getpass

import pandas as pd
from datasets import load_dataset
from huggingface_hub import login


def hf_authenticate() -> None:
    """Authenticate for gated dataset access using huggingface_hub.login when needed."""
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)


def load_ami_dataset_with_fallback():
    """Load dataset with cached auth first; prompt login only if gated access fails."""
    try:
        # token=True tells datasets to use cached HF credentials if present.
        return load_dataset("knkarthick/AMI", token=True)
    except Exception as first_error:
        msg = str(first_error).lower()
        auth_related = any(
            key in msg for key in ["401", "403", "unauthorized", "forbidden", "gated", "access"]
        )
        if not auth_related:
            raise

        token = os.getenv("HF_TOKEN")
        if not token:
            token = getpass("Hugging Face token required for gated AMI dataset: ").strip()

        if not token:
            raise ValueError("No Hugging Face token provided. Set HF_TOKEN or enter token interactively.")

        login(token=token)
        return load_dataset("knkarthick/AMI", token=True)


def normalize_text(text: str) -> str:
    """Clean dialogue/action text and normalize spacing/encoding artifacts."""
    if text is None:
        return ""

    text = str(text)

    # Common mojibake and unicode punctuation cleanup often seen in mixed Taglish data.
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\ufeff": "",
        "\u200b": "",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "Ã±": "ñ",
        "Ã‘": "Ñ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove speaker tags at line starts, e.g. "Speaker 1:", "A:", "B:", "John:"
    text = re.sub(
        r"(?im)^\s*(?:speaker\s*\d+|[A-Z]|[A-Za-z][A-Za-z0-9_\- ]{0,30})\s*:\s*",
        "",
        text,
    )

    # Remove common non-verbal artifacts enclosed in brackets.
    text = re.sub(
        r"(?i)\[(?:laughter|laughs?|uhm+|umm+|unintelligible|inaudible|noise|silence|crosstalk)\]",
        " ",
        text,
    )

    # Collapse extra whitespace/newlines to paragraph form.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_segments(text: str, max_words: int = 40) -> list[str]:
    """Split long text into sentence/phrase-sized segments for cleaner labeling."""
    if not text:
        return []

    sentence_like = re.split(r"(?<=[.!?])\s+", text)
    segments = []

    for piece in sentence_like:
        piece = piece.strip(" \t\n\r\"'")
        if not piece:
            continue

        words = piece.split()
        if len(words) <= max_words:
            segments.append(piece)
            continue

        # If a sentence is very long, split further by soft phrase boundaries.
        phrase_like = re.split(r"\s*(?:,|;|:|\band\b|\bbut\b|\bso\b|\bbecause\b)\s+", piece, flags=re.IGNORECASE)
        for phrase in phrase_like:
            phrase = phrase.strip(" \t\n\r\"'")
            if not phrase:
                continue
            segments.append(phrase)

    # Drop very short noise fragments.
    cleaned = [s for s in segments if len(s) >= 8 and len(s.split()) >= 2]
    return cleaned


def value_to_text(value) -> str:
    """Convert scalars/lists into a clean single text blob."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return " ".join(parts).strip()

    return str(value).strip()


def first_non_empty(record: dict, candidates: list[str]) -> str:
    """Return the first non-empty candidate field from a record."""
    for col in candidates:
        value = value_to_text(record.get(col))
        if value:
            return value
    return ""


def detect_available_splits(dataset_dict) -> list[str]:
    """Find train/validation/test splits even when naming differs (val/dev/eval)."""
    split_keys = list(dataset_dict.keys())
    selected = []

    split_aliases = {
        "train": {"train"},
        "validation": {"validation", "val", "valid", "dev", "eval"},
        "test": {"test"},
    }

    for _, aliases in split_aliases.items():
        found = next((k for k in split_keys if k.lower() in aliases), None)
        if found:
            selected.append(found)

    if not selected:
        raise ValueError(f"No train/validation/test-like splits found. Available splits: {split_keys}")

    return selected


def build_outputs(dataset_dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dialogues and action/info outputs across train/validation/test splits."""
    selected_splits = detect_available_splits(dataset_dict)

    dialogue_rows = []
    labeled_rows = []

    transcript_candidates = ["transcript", "dialogue", "conversation", "meeting_transcript"]
    action_candidates = ["action_item", "action_items", "action", "todo", "task"]
    info_candidates = ["summary", "information", "info", "context_summary", "meeting_summary"]

    for split_name in selected_splits:
        split = dataset_dict[split_name]

        for record in split:
            transcript = normalize_text(first_non_empty(record, transcript_candidates))
            if transcript:
                dialogue_rows.append({"dialogue_text": transcript})

            action_text = normalize_text(first_non_empty(record, action_candidates))
            if action_text:
                for segment in split_into_segments(action_text):
                    labeled_rows.append({"text": segment, "label": 1})

            info_text = normalize_text(first_non_empty(record, info_candidates))
            if info_text:
                for segment in split_into_segments(info_text):
                    labeled_rows.append({"text": segment, "label": 0})

    dialogues_df = pd.DataFrame(dialogue_rows, columns=["dialogue_text"])
    labels_df = pd.DataFrame(labeled_rows, columns=["text", "label"])

    # Remove empty rows and exact duplicates after cleaning.
    dialogues_df = dialogues_df[dialogues_df["dialogue_text"].str.len() > 0].drop_duplicates().reset_index(drop=True)
    labels_df = labels_df[labels_df["text"].str.len() > 0].drop_duplicates().reset_index(drop=True)

    return dialogues_df, labels_df


def main() -> None:
    hf_authenticate()

    ds = load_ami_dataset_with_fallback()
    dialogues_df, labels_df = build_outputs(ds)

    out_dir = os.path.dirname(__file__)
    dialogues_path = os.path.join(out_dir, "ami_dialogues_paragraphs_merged.csv")
    labels_path = os.path.join(out_dir, "ami_action_items_info_merged.csv")

    dialogues_df.to_csv(dialogues_path, index=False, encoding="utf-8")
    labels_df.to_csv(labels_path, index=False, encoding="utf-8")

    print(f"Saved dialogues file: {dialogues_path} | rows={len(dialogues_df)}")
    # action-item & information here are inaccurately labeled as 1 and 0 respectively
    # fix via step by step model training.
    # print(f"Saved action/info file: {labels_path} | rows={len(labels_df)}")


if __name__ == "__main__":
    main()
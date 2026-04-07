"""
Phase 4: Summarization Module
Handles abstractive summarization using BART or Groq API.
"""

from transformers import BartForConditionalGeneration, BartTokenizer

try:
    from integrations.groq.summarize import summarize_with_groq
    GROQ_AVAILABLE = True
    GROQ_IMPORT_ERROR = None
except Exception as e:
    GROQ_AVAILABLE = False
    GROQ_IMPORT_ERROR = str(e)


class LocalSummarizer:
    """Uses BART for local abstractive summarization."""

    def __init__(self):
        """Initialize the BART model and tokenizer."""
        print("Loading BART Summarization Model...")
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

    def generate_summary(self, text, action_items=None):
        """
        Generate a summary using BART.

        Args:
            text (str): Meeting transcript or context text
            action_items (list): List of detected action items

        Returns:
            str: Generated summary
        """
        input_text = f"Meeting: {text}"
        if action_items:
            input_text += " Tasks: " + " . ".join(action_items)

        inputs = self.tokenizer(
            [input_text], max_length=1024, return_tensors="pt", truncation=True
        )

        # Generate summary with optimized parameters
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=10,
            max_length=60,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            early_stopping=True,
            forced_bos_token_id=0,
        )

        decoded_summary = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )

        # Post-processing: Remove prompt if copied by model
        if "Summarize the" in decoded_summary:
            decoded_summary = decoded_summary.split("list.")[-1].strip()

        return decoded_summary


class Summarizer:
    """Orchestrates summarization using either Groq or Local BART."""

    def __init__(self, engine="local"):
        """
        Initialize the summarizer.

        Args:
            engine (str): "local" for BART or "groq" for Groq API
        """
        self.engine = engine
        self.local_summarizer = None
        self.groq_available = GROQ_AVAILABLE

        if engine == "local":
            self.local_summarizer = LocalSummarizer()

    def generate_summary(self, text, action_items=None):
        """
        Generate a summary using the configured engine.

        Args:
            text (str): Meeting transcript or context text
            action_items (list): List of detected action items

        Returns:
            str: Generated summary
        """
        if self.engine == "groq":
            if not self.groq_available:
                print(f"[System] Groq summarizer unavailable: {GROQ_IMPORT_ERROR}")
                print("[System] Falling back to local BART summarizer...")
                if self.local_summarizer is None:
                    self.local_summarizer = LocalSummarizer()
                return self.local_summarizer.generate_summary(text, action_items)

            try:
                return summarize_with_groq(text, action_items)
            except Exception as e:
                print(f"[System] Groq summary failed: {e}")
                print("[System] Falling back to local BART summarizer...")
                if self.local_summarizer is None:
                    self.local_summarizer = LocalSummarizer()
                return self.local_summarizer.generate_summary(text, action_items)

        # Local BART
        if self.local_summarizer is None:
            self.local_summarizer = LocalSummarizer()
        return self.local_summarizer.generate_summary(text, action_items)

    def switch_engine(self, engine):
        """
        Switch between engines at runtime.

        Args:
            engine (str): "local" or "groq"
        """
        self.engine = engine
        if engine == "local" and self.local_summarizer is None:
            self.local_summarizer = LocalSummarizer()

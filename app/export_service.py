import os
import re
import json
from datetime import datetime
from html import escape
from tempfile import NamedTemporaryFile


class ReportContentFormatter:
    """Text normalization utilities for PDF-friendly output."""

    SUMMARY_LABEL_PATTERN = re.compile(
        r"^(meeting summary|executive summary|summary|action items?)\s*[:\-]*\s*",
        flags=re.IGNORECASE,
    )

    MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s*")
    BULLET_PREFIX_PATTERN = re.compile(r"^\s*[-*•]\s+")
    ACTION_SECTION_PATTERN = re.compile(r"^\s*(?:#{1,6}\s*)?action items?\s*:?\s*$", flags=re.IGNORECASE)
    NUMBERED_PREFIX_PATTERN = re.compile(r"^\s*\d+[\.)]\s+")
    SYSTEM_LINE_PATTERN = re.compile(r"^\s*\[(?:system|speaker\s*\d+|speaker|timestamp)\][\s:.-]*", flags=re.IGNORECASE)
    TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\s*\[(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}:\d{2})\]\s*")

    def __init__(self):
        self._llama_explanation_cache = {}

    def clean_summary(self, summary_text):
        lines = (summary_text or "").splitlines()
        cleaned_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            line = self.MARKDOWN_HEADING_PATTERN.sub("", line)
            line = line.replace("**", "").replace("__", "").replace("`", "")
            line = self.BULLET_PREFIX_PATTERN.sub("", line)

            # Remove leading labels to avoid showing "Meeting Summary" or "Action Items" in narrative.
            normalized = self.SUMMARY_LABEL_PATTERN.sub("", line).strip()
            if not normalized:
                continue

            lower = normalized.lower()
            if lower.startswith("action item") or lower.startswith("action items"):
                continue

            cleaned_lines.append(normalized)

        merged = " ".join(cleaned_lines)
        merged = re.sub(r"\s+", " ", merged).strip()
        # Drop trailing action-item block if model appended it in-line.
        merged = re.sub(r"\baction items?\b\s*[:\-].*$", "", merged, flags=re.IGNORECASE).strip()
        return merged

    def split_summary_into_paragraphs(self, summary_text, duration_seconds=None):
        text = self.clean_summary(summary_text)
        if not text:
            # Generate duration-aware placeholder message
            if duration_seconds:
                if duration_seconds < 300:  # Less than 5 minutes
                    return ["This was a brief meeting covering the main discussion points and decisions."]
                elif duration_seconds < 1800:  # Less than 30 minutes
                    return ["This meeting addressed key topics with focused discussion and identified action items."]
                else:
                    return ["This comprehensive meeting covered multiple topics in depth, with thorough discussion and clear next steps."]
            return ["No executive summary was generated for this session."]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return [text]

        # Adjust paragraph count based on meeting duration and content length
        if duration_seconds:
            if duration_seconds < 300:  # Less than 5 minutes
                target_paragraphs = 1
            elif duration_seconds < 1800:  # Less than 30 minutes
                target_paragraphs = 2
            else:
                target_paragraphs = 3
        else:
            target_paragraphs = 3 if len(sentences) >= 9 else 2
        
        chunk_size = max(1, len(sentences) // target_paragraphs)
        paragraphs = []

        start = 0
        for idx in range(target_paragraphs):
            end = len(sentences) if idx == target_paragraphs - 1 else min(len(sentences), start + chunk_size)
            if start >= len(sentences):
                break
            paragraph = " ".join(sentences[start:end]).strip()
            if paragraph:
                paragraphs.append(paragraph)
            start = end

        return paragraphs if paragraphs else [text]

    def build_action_explanation(self, action_item):
        """Generate a contextual explanation for an action item using Llama."""
        item = (action_item or "").strip()
        if not item:
            return ""

        # If the item is too vague, do not show a task description at all.
        if self._action_item_needs_more_context(item):
            return ""

        if item in self._llama_explanation_cache:
            return self._llama_explanation_cache[item]

        explanation = self._generate_llama_explanation(item)
        explanation = self._normalize_single_sentence(explanation)

        if not explanation or self._action_item_needs_more_context(explanation):
            explanation = ""

        self._llama_explanation_cache[item] = explanation
        return explanation

    def _normalize_single_sentence(self, text):
        """Keep only the first sentence and remove excess whitespace."""
        value = (text or "").strip()
        if not value:
            return ""

        value = re.sub(r"\s+", " ", value).strip()
        sentence_match = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)
        value = sentence_match[0].strip() if sentence_match else value
        if value and value[-1] not in ".!?":
            value += "."
        return value

    def _action_item_needs_more_context(self, action_item):
        """Return True when an item is too vague to safely explain."""
        item = re.sub(r"\s+", " ", (action_item or "")).strip().lower()
        if not item:
            return True

        vague_patterns = [
            r"^submit\b(?:\s+ang)?\s+deadline\b.*$",
            r"^gawin\b.*$",
            r"^do\b.*$",
            r"^task\b.*$",
            r"^follow\s*up\b.*$",
            r"^update\b.*$",
            r"^review\b.*$",
        ]
        if any(re.fullmatch(pattern, item) for pattern in vague_patterns):
            return True

        # Very short items usually lack enough context to describe safely.
        if len(item.split()) <= 4:
            return True

        return False
    
    def _generate_llama_explanation(self, action_item):
        """Call Groq/Llama to generate a brief action task explanation."""
        try:
            from integrations.groq.summarize import summarize_with_groq
            import os
            from dotenv import load_dotenv
        except ImportError:
            # Fallback if imports fail
            return self._fallback_action_explanation(action_item)
        
        try:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return self._fallback_action_explanation(action_item)
            
            from groq import Groq
        except (ImportError, RuntimeError):
            return self._fallback_action_explanation(action_item)
        
        try:
            client = Groq(api_key=api_key)
            model = os.getenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant")
            
            prompt = (
                f"Generate exactly one sentence for the following action item. "
                f"If the action item is too vague or lacks enough context, return an empty string. "
                f"Do not invent details, do not add assumptions, and do not include quotation marks or extra formatting.\n\n"
                f"Action Item: {action_item}"
            )
            
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task management assistant. Provide clear, concise action task descriptions."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=100,
            )
            
            explanation = result.choices[0].message.content.strip() if result.choices else None
            
            if explanation and len(explanation) > 0:
                # Clean up any quotes if present
                explanation = explanation.strip('\'"')
                explanation = self._normalize_single_sentence(explanation)
                return explanation
            else:
                return ""
                
        except Exception as e:
            # Log error silently and fall back to static explanation
            print(f"Groq API error generating explanation: {e}")
            return ""
    
    def _fallback_action_explanation(self, action_item):
        """Fallback static explanation based on action verb."""
        item = (action_item or "").strip()
        item_lower = item.lower()
        
        # Extract key action verbs and nouns for personalization
        if any(verb in item_lower for verb in ["prepare", "create", "write", "draft", "develop"]):
            return f"Take ownership of {item}. Set a clear deadline, break it into smaller milestones, and share progress weekly with stakeholders."
        
        if any(verb in item_lower for verb in ["review", "analyze", "audit", "check", "examine"]):
            return f"Schedule time to {item_lower.split()[0] if item_lower.split() else 'review'} thoroughly. Document your findings and share feedback with the team."
        
        if any(verb in item_lower for verb in ["schedule", "organize", "arrange", "book", "set up"]):
            return f"Coordinate {item} with all participants. Send calendar invites, attach agenda, and confirm attendance 24 hours prior."
        
        if any(verb in item_lower for verb in ["test", "validate", "verify", "qa"]):
            return f"Execute {item} in a controlled environment. Log all findings, prioritize issues, and retest after fixes are applied."
        
        if any(verb in item_lower for verb in ["submit", "deliver", "send", "provide", "share"]):
            return f"Complete {item} by the agreed deadline. Ensure quality, obtain any required approvals, and deliver to the specified recipients."
        
        if any(verb in item_lower for verb in ["update", "document", "record", "log"]):
            return f"Keep {item} current and accurate. Update the team regularly and ensure all relevant stakeholders have access."
        
        # Fallback: generic actionable task
        return f"Assign ownership, set a clear deadline, track progress, and communicate status updates regularly. Document milestones and ensure team accountability."

    def extract_llama_action_items(self, summary_text):
        """Extract bullet/numbered action items produced by Llama in the summary output."""
        lines = (summary_text or "").splitlines()
        in_action_section = False
        items = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if in_action_section:
                    continue
                continue

            if self.ACTION_SECTION_PATTERN.match(line):
                in_action_section = True
                continue

            if not in_action_section:
                # Handle inline format: "Action Items: - item one"
                inline_match = re.match(r"^\s*(?:#{1,6}\s*)?action items?\s*:\s*(.+)$", line, flags=re.IGNORECASE)
                if inline_match:
                    in_action_section = True
                    line = inline_match.group(1).strip()
                else:
                    continue

            # If section has started, capture bullet/numbered lines.
            candidate = self.BULLET_PREFIX_PATTERN.sub("", line)
            candidate = self.NUMBERED_PREFIX_PATTERN.sub("", candidate).strip()

            # Stop if a new major heading appears after action section started.
            if re.match(r"^\s*(?:#{1,6}\s*)?[A-Za-z][A-Za-z\s]{2,30}:\s*$", line):
                break

            if not candidate:
                continue

            lowered = candidate.lower()
            if any(marker in lowered for marker in ["none identified", "no action items", "none provided", "n/a"]):
                continue

            items.append(candidate)

        deduped = []
        seen = set()
        for item in items:
            key = re.sub(r"\s+", " ", item.strip().lower())
            if key and key not in seen:
                seen.add(key)
                deduped.append(item.strip())

        return deduped

    def clean_transcript_text(self, transcript_text):
        """Remove transcript metadata lines and timestamp/speaker markers."""
        lines = (transcript_text or "").splitlines()
        cleaned_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if self.SYSTEM_LINE_PATTERN.match(line):
                continue

            if self.TIMESTAMP_PREFIX_PATTERN.match(line):
                line = self.TIMESTAMP_PREFIX_PATTERN.sub("", line).strip()

            if re.fullmatch(r"\[(?:speaker\s*\d+|speaker|system)\]", line, flags=re.IGNORECASE):
                continue

            if not line:
                continue

            cleaned_lines.append(line)

        cleaned_text = " ".join(cleaned_lines)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def build_action_flag_reason(self, action_item):
        """Return a conservative explanation for why an action item was flagged."""
        item = (action_item or "").strip()
        if not item:
            return "No action text was available for analysis."

        lower = item.lower()
        if any(marker in lower for marker in ["please", "pwede", "paki", "pakis", "can you", "could you", "need to", "should", "must"]):
            return "Flagged because the utterance contains a request or directive."

        if any(marker in lower for marker in ["due", "deadline", "by ", "before ", "deliver", "submit", "send", "prepare", "review"]):
            return "Flagged because it describes a task or follow-up that implies completion."

        return "Flagged because it matches the classifier's action-item pattern without adding assumptions."


class ReportStyleFactory:
    """Creates a centralized style map to keep PDF layout easy to maintain."""

    @staticmethod
    def build(styles, colors, ParagraphStyle, TA_JUSTIFY):
        return {
            "title": ParagraphStyle(
                "MainTitle",
                parent=styles["Title"],
                fontSize=21,
                leading=25,
                spaceAfter=10,
                textColor=colors.HexColor("#1F3B57"),
            ),
            "heading": ParagraphStyle(
                "SectionHeader",
                parent=styles["Heading2"],
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#1B5F8A"),
                spaceBefore=10,
                spaceAfter=5,
            ),
            "body": ParagraphStyle(
                "MeetingBody",
                parent=styles["Normal"],
                fontSize=10.5,
                leading=15,
                alignment=TA_JUSTIFY,
                spaceAfter=8,
            ),
            "meta": ParagraphStyle(
                "Meta",
                parent=styles["Normal"],
                fontSize=9.5,
                leading=13,
                textColor=colors.HexColor("#3F4B57"),
                spaceAfter=2,
            ),
            "action_item": ParagraphStyle(
                "ActionItem",
                parent=styles["Normal"],
                fontSize=10.5,
                leading=14,
                leftIndent=14,
                bulletIndent=4,
                spaceAfter=2,
            ),
            "action_help": ParagraphStyle(
                "ActionHelp",
                parent=styles["Normal"],
                fontSize=9.5,
                leading=13,
                leftIndent=28,
                textColor=colors.HexColor("#4A5560"),
                spaceAfter=7,
            ),
            "transcript": ParagraphStyle(
                "Transcript",
                parent=styles["Normal"],
                fontSize=10.5,
                leading=15,
                alignment=TA_JUSTIFY,
                spaceAfter=8,
            ),
        }


class ExportService:
    DEFAULT_SECTIONS = [
        "Executive Overview",
        "Topics Discussed",
        "Action Items",
        "Full Transcript",
    ]

    def __init__(self):
        self.output_dir = os.path.join("output", "pdf")
        self.formatter = ReportContentFormatter()

    def _extract_json_object(self, text):
        """Extract first JSON object from model output, including fenced output."""
        value = (text or "").strip()
        if not value:
            return None

        if value.startswith("```"):
            value = value.strip("`").strip()
            if value.lower().startswith("json"):
                value = value[4:].lstrip()

        start = value.find("{")
        end = value.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = value[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def extract_action_items_fast(self, transcript_text, max_items=25):
        """Fast Llama-first extractor that returns only transcript-grounded action phrases."""
        clean_text = self.formatter.clean_transcript_text(transcript_text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        details = []

        def trim_action_phrase(text_value):
            text_value = re.sub(r"\s+", " ", (text_value or "")).strip()
            if not text_value:
                return ""

            # Split on common discourse separators and keep the chunk with strongest task cue.
            chunks = [c.strip(" ,;:-") for c in re.split(r"\b(?:and then|then|at saka|pero|but|however)\b", text_value, flags=re.IGNORECASE) if c.strip()]
            if len(chunks) <= 1:
                return text_value

            cue_pattern = re.compile(
                r"\b(please|pwede|paki|can you|could you|need to|must|should|due|submit|send|prepare|review|asikasuhin)\b",
                flags=re.IGNORECASE,
            )
            scored = sorted(chunks, key=lambda c: (1 if cue_pattern.search(c) else 0, len(c)), reverse=True)
            return scored[0] if scored else text_value

        # Regex fallback (also used as safety net when model output is invalid).
        def fallback_candidates():
            markers = [
                r"\bplease\b",
                r"\bpwede\b",
                r"\bpaki\b",
                r"\bcan you\b",
                r"\bcould you\b",
                r"\bneed to\b",
                r"\bmust\b",
                r"\bshould\b",
                r"\bdue\b",
                r"\bsubmit\b",
                r"\bsend\b",
                r"\bprepare\b",
                r"\breview\b",
            ]
            pattern = re.compile("|".join(markers), flags=re.IGNORECASE)
            out = []
            for sentence in sentences:
                if pattern.search(sentence):
                    out.append(sentence)
            return out

        model_items = []
        try:
            from dotenv import load_dotenv
            from groq import Groq

            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                client = Groq(api_key=api_key)
                model = os.getenv("GROQ_MODEL_LLAMA", "llama-3.1-8b-instant")

                system_prompt = (
                    "You extract action items from meeting transcripts. "
                    "Return JSON only with key 'items'. Each item must have keys 'text' and 'why'. "
                    "CRITICAL: 'text' must be an exact substring from the transcript and should contain only the action phrase/sentence, "
                    "not surrounding greetings or unrelated discussion. "
                    "If uncertain, exclude it."
                )
                user_prompt = (
                    "Transcript:\n"
                    f"{clean_text[:12000]}\n\n"
                    "Return format:\n"
                    '{"items":[{"text":"...","why":"..."}]}'
                )

                result = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_completion_tokens=500,
                )

                content = result.choices[0].message.content if result.choices else ""
                payload = self._extract_json_object(content)
                items = payload.get("items", []) if isinstance(payload, dict) else []
                for item in items:
                    text_value = str((item or {}).get("text", "")).strip()
                    why_value = str((item or {}).get("why", "")).strip()
                    if text_value:
                        model_items.append({"text": text_value, "why": why_value})
        except Exception:
            model_items = []

        if not model_items:
            model_items = [{"text": sentence, "why": "Matched request/task pattern."} for sentence in fallback_candidates()]

        # Enforce transcript-grounded extraction and deduplicate.
        lowered_clean = clean_text.lower()
        seen = set()
        action_items = []
        for item in model_items:
            text_value = str(item.get("text", "")).strip()
            if not text_value:
                continue

            text_value = trim_action_phrase(text_value)

            # Keep only direct transcript matches to avoid fabricated wording.
            if text_value.lower() not in lowered_clean:
                continue

            norm = re.sub(r"\s+", " ", text_value.lower())
            if norm in seen:
                continue
            seen.add(norm)

            action_items.append(text_value)
            details.append(
                {
                    "item": text_value,
                    "reason": str(item.get("why", "")).strip() or "Contains an explicit task/request cue.",
                }
            )
            if len(action_items) >= max_items:
                break

        return {
            "action_items": action_items,
            "details": details,
            "total_sentences": len(sentences),
            "clean_transcript": clean_text,
        }

    def build_transcript_analytics(self, transcript_text, action_items=None, sentences=None):
        """Build conservative analytics for the transcript without inventing labels."""
        action_items = [str(item).strip() for item in (action_items or []) if str(item).strip()]
        transcript_text = transcript_text or ""

        if sentences is None:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript_text) if s.strip()]

        total_sentences = len(sentences)
        action_count = len(action_items)
        info_count = max(0, total_sentences - action_count)

        def _extract_keywords_with_weights(items, top_n=5):
            tokens = []
            stop_words = {"please", "paki", "pwede", "bang", "ng", "mga", "the", "and", "for", "with", "that", "this", "will", "team", "you", "your"}
            action_cues = {
                "submit", "send", "prepare", "review", "complete", "finalize", "update",
                "coordinate", "call", "follow", "draft", "deliver", "share", "check",
                "sara", "bukas", "tawag", "ilipat", "tulong", "asikaso", "ayos", "gawa",
            }

            for item in items:
                words = re.findall(r"[A-Za-zÀ-ÿ']+", item.lower())
                for word in words:
                    if len(word) < 4:
                        continue
                    if word in stop_words:
                        continue
                    # Keep action-like words first; if none are found we will fallback below.
                    if word in action_cues:
                        tokens.append(word)

            # Fallback to generic keyword extraction when action-cue-only set is empty.
            if not tokens:
                for item in items:
                    words = re.findall(r"[A-Za-zÀ-ÿ']+", item.lower())
                    for word in words:
                        if len(word) < 4:
                            continue
                        if word in stop_words:
                            continue
                        tokens.append(word)

            from collections import Counter
            counter = Counter(tokens)
            total = sum(counter.values()) or 1
            top = counter.most_common(top_n)
            # return list of (word, count, weight_float)
            tuples = [(w, c, float(c) / float(total)) for w, c in top]
            rows = [
                {
                    "keyword": w,
                    "count": int(c),
                    "total_tokens": int(total),
                    "weight_score": float(wt),
                    "weight_percent": float(wt * 100.0),
                    "formula": f"{c}/{total}",
                }
                for w, c, wt in tuples
            ]
            return tuples, int(sum(counter.values())), rows

        weighted_keywords, total_keyword_tokens, weight_rows = _extract_keywords_with_weights(action_items)

        return {
            "total_sentences": total_sentences,
            "action_count": action_count,
            "info_count": info_count,
            "action_ratio": (action_count / total_sentences) if total_sentences else 0.0,
            "info_ratio": (info_count / total_sentences) if total_sentences else 0.0,
            "top_action_keywords": weighted_keywords,
            "action_keyword_total_tokens": total_keyword_tokens,
            "top_action_keyword_rows": weight_rows,
            "action_items": action_items,
        }

    def build_preview_summary(self, transcript_text, action_items=None, summary_text=None):
        """Build an executive overview for preview/PDF export.

        Prefer an explicit summary when provided, then Groq/Llama summarization, then a short
        transcript excerpt fallback.
        """
        summary_text = (summary_text or "").strip()
        if summary_text:
            return summary_text

        clean_text = self.formatter.clean_transcript_text(transcript_text)
        if not clean_text:
            return "No executive summary was generated for this session."

        try:
            from integrations.groq.summarize import summarize_with_groq

            generated = summarize_with_groq(clean_text, action_items or [])
            generated = (generated or "").strip()
            if generated:
                generated = re.split(r"\n\s*Action Items\s*:", generated, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                return self.formatter.clean_summary(generated) or generated
        except Exception:
            pass

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        if sentences:
            return " ".join(sentences[:2])
        return clean_text[:300]

    def build_topic_labels(self, transcript_text, max_topics=5):
        """Build topic labels for preview/PDF export.

        Prefer the semantic segmenter when available, then fall back to short transcript-derived labels.
        """
        clean_text = self.formatter.clean_transcript_text(transcript_text)
        if not clean_text:
            return []

        try:
            from core.segmenter import Segmenter

            segmenter = Segmenter(chunk_size=5, max_tokens=250)
            segmenter.segment_text(clean_text)
            topics = []
            seen = set()
            for segment in segmenter.get_segment_metadata():
                desc = str(segment.get("topical_description", "")).strip()
                if not desc:
                    continue
                normalized = re.sub(r"\s+", " ", desc.lower())
                if normalized in seen:
                    continue
                seen.add(normalized)
                topics.append(desc[:80] + ("..." if len(desc) > 80 else ""))
                if len(topics) >= max_topics:
                    break
            if topics:
                return topics
        except Exception:
            pass

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        topics = []
        for sentence in sentences:
            words = re.findall(r"[A-Za-zÀ-ÿ']+", sentence)
            if len(words) < 4:
                continue
            topic = " ".join(words[:8]).strip()
            normalized = re.sub(r"\s+", " ", topic.lower())
            if normalized and normalized not in {re.sub(r"\s+", " ", t.lower()) for t in topics}:
                topics.append(topic)
            if len(topics) >= max_topics:
                break

        return topics

    def _build_analytics_chart(self, analytics):
        """Render a pie chart for action vs information items and return a PNG path."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except Exception:
            return None

        action_count = int(analytics.get("action_count", 0))
        info_count = int(analytics.get("info_count", 0))

        if action_count == 0 and info_count == 0:
            return None

        # Create combined figure: left doughnut (action vs info), right horizontal bars for top keywords
        top_keywords = analytics.get("top_action_keywords", []) or []

        fig = plt.figure(figsize=(7.0, 2.8), dpi=160)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.3)

        # Left: doughnut pie
        ax0 = fig.add_subplot(gs[0])
        values = [max(action_count, 0), max(info_count, 0)]
        labels = ["Action Items", "Information Items"]
        colors_list = ["#e04f4f", "#3b82f6"]
        wedges, texts, autotexts = ax0.pie(
            values,
            labels=labels,
            autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
            startangle=90,
            colors=colors_list,
            textprops={"color": "#0f1720", "fontsize": 9},
            wedgeprops={"width": 0.45, "edgecolor": "w"},
        )
        ax0.set_aspect("equal")
        ax0.set_title("Transcript Breakdown", fontsize=11, pad=8)

        # Center text: total segments/sentences
        total_label = analytics.get("total_sentences", 0)
        ax0.text(0, 0, f"{total_label}\nsegments", ha="center", va="center", fontsize=10, color="#0f1720")

        # Right: horizontal bars for keywords (if present)
        ax1 = fig.add_subplot(gs[1])
        if top_keywords:
            words = [w for w, c, wt in top_keywords]
            counts = [c for w, c, wt in top_keywords]
            weights = [wt for w, c, wt in top_keywords]
            # Display from top to bottom
            y_pos = list(range(len(words)))[::-1]
            ax1.barh(y_pos, counts[::-1], color="#a45bd6")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(words[::-1], fontsize=9)
            ax1.invert_yaxis()
            ax1.xaxis.set_visible(False)
            ax1.set_xlim(0, max(1, max(counts) * 1.3))
            # Annotate counts and percentages
            for i, (cnt, wt) in enumerate(zip(counts[::-1], weights[::-1])):
                ax1.text(cnt + (max(counts) * 0.02), i, f"{cnt}  ({int(round(wt*100))}%)", va="center", fontsize=9, color="#0f1720")
            ax1.set_title("Action Keyword Frequency", fontsize=11, pad=8)
        else:
            ax1.text(0.5, 0.5, "No recurring action keywords detected.", ha="center", va="center", fontsize=10, color="#6b7280")
            ax1.set_axis_off()

        temp_file = NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = temp_file.name
        temp_file.close()
        fig.savefig(temp_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return temp_path

    def _build_separate_analytics_charts(self, analytics):
        """Render two separate charts: a doughnut for breakdown and a horizontal bar chart for keywords.
        Returns tuple (breakdown_png_path, keywords_png_path) where either may be None on failure.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return (None, None)

        action_count = int(analytics.get("action_count", 0))
        info_count = int(analytics.get("info_count", 0))

        # Breakdown doughnut
        try:
            fig1, ax1 = plt.subplots(figsize=(3.0, 2.8), dpi=160)
            values = [max(action_count, 0), max(info_count, 0)]
            labels = ["Action Items", "Information Items"]
            colors_list = ["#e04f4f", "#3b82f6"]
            ax1.pie(
                values,
                labels=labels,
                autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
                startangle=90,
                colors=colors_list,
                textprops={"color": "#0f1720", "fontsize": 9},
                wedgeprops={"width": 0.45, "edgecolor": "w"},
            )
            ax1.set_aspect("equal")
            ax1.set_title("Transcript Breakdown", fontsize=11, pad=8)
            total_label = analytics.get("total_sentences", 0)
            ax1.text(0, 0, f"{total_label}\nsegments", ha="center", va="center", fontsize=10, color="#0f1720")

            tmp1 = NamedTemporaryFile(delete=False, suffix=".png")
            p1 = tmp1.name
            tmp1.close()
            fig1.savefig(p1, bbox_inches="tight", facecolor="white")
            plt.close(fig1)
        except Exception:
            p1 = None

        # Keywords horizontal bars
        try:
            fig2, ax2 = plt.subplots(figsize=(4.0, 2.8), dpi=160)
            top_keywords = analytics.get("top_action_keywords", []) or []
            if top_keywords:
                words = [w for w, c, wt in top_keywords]
                counts = [c for w, c, wt in top_keywords]
                weights = [wt for w, c, wt in top_keywords]
                y_pos = list(range(len(words)))[::-1]
                ax2.barh(y_pos, counts[::-1], color="#a45bd6")
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(words[::-1], fontsize=9)
                ax2.invert_yaxis()
                ax2.xaxis.set_visible(False)
                ax2.set_xlim(0, max(1, max(counts) * 1.3))
                for i, (cnt, wt) in enumerate(zip(counts[::-1], weights[::-1])):
                    ax2.text(cnt + (max(counts) * 0.02), i, f"{cnt}  ({int(round(wt*100))}%)", va="center", fontsize=9, color="#0f1720")
                ax2.set_title("Action Keyword Frequency", fontsize=11, pad=8)
            else:
                ax2.text(0.5, 0.5, "No recurring action keywords detected.", ha="center", va="center", fontsize=10, color="#6b7280")
                ax2.set_axis_off()

            tmp2 = NamedTemporaryFile(delete=False, suffix=".png")
            p2 = tmp2.name
            tmp2.close()
            fig2.savefig(p2, bbox_inches="tight", facecolor="white")
            plt.close(fig2)
        except Exception:
            p2 = None

        return (p1, p2)

    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_filename(self, name):
        cleaned = re.sub(r'[<>:"/\\|?*]', "", (name or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned or "Meeting"

    def _build_pdf_path(self, source_file):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        meeting_prefix = f"Meeting_Minutes_{timestamp}"

        if source_file:
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            safe_base = self._sanitize_filename(base_name)
            file_stem = f"{meeting_prefix}_{safe_base}"
        else:
            file_stem = meeting_prefix

        pdf_path = os.path.join(self.output_dir, f"{file_stem}.pdf")

        if not os.path.exists(pdf_path):
            return pdf_path

        # Keep the requested naming format and avoid overwriting existing exports.
        counter = 2
        while True:
            candidate = os.path.join(self.output_dir, f"{file_stem} ({counter}).pdf")
            if not os.path.exists(candidate):
                return candidate
            counter += 1

    def generate_pdf(
        self,
        content,
        action_items=None,
        summary=None,
        duration_seconds=None,
        start_time=None,
        end_time=None,
        source_file=None,
        topics=None,
        section_order=None,
        include_sections=None,
    ):
        """Generate a professional PDF report with narrative summary and Llama action items."""
        clean_content = self.formatter.clean_transcript_text(content)
        analytics = self.build_transcript_analytics(clean_content, action_items=action_items)
        chart_path = None

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Image as RLImage
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_JUSTIFY
        except ModuleNotFoundError as e:
            raise RuntimeError("Missing dependency: reportlab. Run: pip install reportlab") from e

        self._ensure_output_dir()

        # Keep only file name (not full source path) in the header.
        file_display_name = os.path.basename(source_file) if source_file else "Live Recording"

        duration_str = self._format_duration(duration_seconds, start_time, end_time)

        summary_paragraphs = self.formatter.split_summary_into_paragraphs(summary or "", duration_seconds)

        order = [section for section in (section_order or self.DEFAULT_SECTIONS) if section in self.DEFAULT_SECTIONS]
        for section in self.DEFAULT_SECTIONS:
            if section not in order:
                order.append(section)

        selected_sections = set(include_sections or self.DEFAULT_SECTIONS)

        selected_action_items = [str(item).strip() for item in (action_items or []) if str(item).strip()]
        if not selected_action_items:
            selected_action_items = self.formatter.extract_llama_action_items(summary or "")

        pdf_path = self._build_pdf_path(source_file)

        styles = getSampleStyleSheet()
        report_styles = ReportStyleFactory.build(styles, colors, ParagraphStyle, TA_JUSTIFY)

        # Try to create separate analytics images (breakdown + keywords). Fall back to combined chart.
        try:
            breakdown_path, keywords_path = self._build_separate_analytics_charts(analytics)
        except Exception:
            breakdown_path, keywords_path = (None, None)
        chart_path = breakdown_path or self._build_analytics_chart(analytics)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=56,
            leftMargin=56,
            topMargin=50,
            bottomMargin=40,
        )
        story = []

        story.append(Paragraph("Minutes of the Meeting", report_styles["title"]))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#A8B5C1"), spaceAfter=8))
        story.append(Paragraph(f"<b>Source File:</b> {escape(file_display_name)}", report_styles["meta"]))
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", report_styles["meta"]))
        story.append(Paragraph(f"<b>Meeting Duration:</b> {duration_str}", report_styles["meta"]))
        story.append(Spacer(1, 0.18 * inch))

        if analytics:
            story.append(Paragraph("Analytics Overview", report_styles["heading"]))
            story.append(
                Paragraph(
                    escape(
                        f"Total sentences detected: {analytics['total_sentences']}. "
                        f"Suggested action items: {analytics['action_count']}. "
                        f"Information sentences: {analytics['info_count']}."
                    ),
                    report_styles["body"],
                )
            )
            if analytics.get("top_action_keywords"):
                # top_action_keywords contains (word, count, weight)
                keyword_text = ", ".join(f"{word} ({count}, {int(round(weight*100))}%)" for word, count, weight in analytics["top_action_keywords"])
                story.append(Paragraph(f"<b>Common action keywords:</b> {escape(keyword_text)}", report_styles["meta"]))

            # Prefer showing separate images side-by-side when available
            if breakdown_path and keywords_path and os.path.exists(breakdown_path) and os.path.exists(keywords_path):
                try:
                    from reportlab.platypus import Table, TableStyle

                    img_left = RLImage(breakdown_path, width=3.0 * inch, height=2.2 * inch)
                    img_right = RLImage(keywords_path, width=3.5 * inch, height=2.2 * inch)
                    tbl = Table([[img_left, img_right]], colWidths=[3.0 * inch, 3.5 * inch])
                    tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
                    story.append(Spacer(1, 0.08 * inch))
                    story.append(tbl)
                    story.append(Spacer(1, 0.12 * inch))
                except Exception:
                    story.append(Spacer(1, 0.08 * inch))
                    if chart_path and os.path.exists(chart_path):
                        story.append(RLImage(chart_path, width=3.5 * inch, height=2.2 * inch))
                        story.append(Spacer(1, 0.12 * inch))
            elif chart_path and os.path.exists(chart_path):
                story.append(Spacer(1, 0.08 * inch))
                story.append(RLImage(chart_path, width=3.5 * inch, height=2.2 * inch))
                story.append(Spacer(1, 0.12 * inch))

        for section in order:
            if section not in selected_sections:
                continue

            if section == "Executive Overview":
                story.append(Paragraph("Executive Overview", report_styles["heading"]))
                for paragraph in summary_paragraphs:
                    story.append(Paragraph(escape(paragraph), report_styles["body"]))
                story.append(Spacer(1, 0.1 * inch))
                continue

            if section == "Topics Discussed":
                story.append(Paragraph("Topics Discussed", report_styles["heading"]))
                topic_list = topics if isinstance(topics, list) else [topics]
                has_topics = False
                for idx, topic in enumerate(topic_list, 1):
                    clean_topic = str(topic).strip() if topic is not None else ""
                    if clean_topic:
                        has_topics = True
                        story.append(Paragraph(f"<b>{idx}.</b> {escape(clean_topic)}", report_styles["body"]))
                if not has_topics:
                    story.append(Paragraph("<i>No topics were identified.</i>", report_styles["body"]))
                story.append(Spacer(1, 0.1 * inch))
                continue

            if section == "Action Items":
                story.append(Paragraph("Action Items", report_styles["heading"]))
                if selected_action_items:
                    for item in selected_action_items:
                        clean_item = item.strip()
                        if not clean_item:
                            continue
                        story.append(Paragraph(f"<b>[ ]</b> {escape(clean_item)}", report_styles["action_item"]))
                        task_text = self.formatter.build_action_explanation(clean_item)
                        if task_text:
                            story.append(
                                Paragraph(
                                    f"<i>Task:</i> {escape(task_text)}",
                                    report_styles["action_help"],
                                )
                            )
                        story.append(
                            Paragraph(
                                f"<i>Why flagged:</i> {escape(self.formatter.build_action_flag_reason(clean_item))}",
                                report_styles["action_help"],
                            )
                        )
                else:
                    story.append(Paragraph("<i>No action items were selected for this report.</i>", report_styles["body"]))
                story.append(Spacer(1, 0.1 * inch))
                continue

            if section == "Full Transcript":
                story.append(Spacer(1, 0.14 * inch))
                story.append(Paragraph("Full Transcript", report_styles["heading"]))
                safe_content = escape(clean_content or "[Empty Transcript]").replace("\n", "<br/>")
                story.append(Paragraph(safe_content, report_styles["transcript"]))

        doc.build(story)
        if chart_path and os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception:
                pass
        return pdf_path

    def _format_duration(self, duration_seconds=None, start_time=None, end_time=None):
        seconds = None
        if duration_seconds is not None:
            seconds = max(0, int(round(duration_seconds)))
        elif start_time and end_time:
            try:
                seconds = max(0, int(round((end_time - start_time).total_seconds())))
            except Exception:
                seconds = None

        if seconds is None:
            return "N/A"

        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

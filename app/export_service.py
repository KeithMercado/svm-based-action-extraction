import os
import re
from datetime import datetime
from html import escape


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

    def split_summary_into_paragraphs(self, summary_text):
        text = self.clean_summary(summary_text)
        if not text:
            return ["No executive summary was generated for this session."]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return [text]

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
        """Generate a contextual, relatable explanation tied to the specific action item."""
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
        
        # Fallback: create a simple, direct explanation
        return f"Complete this action item: {item}. Assign ownership, set a deadline, track progress, and communicate status updates regularly."

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
                fontSize=8.8,
                leading=11,
                textColor=colors.HexColor("#5A6570"),
            ),
        }


class ExportService:
    def __init__(self):
        self.output_dir = os.path.join("output", "pdf")
        self.formatter = ReportContentFormatter()

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
    ):
        """Generate a professional PDF report with narrative summary and Llama action items."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_JUSTIFY
        except ModuleNotFoundError as e:
            raise RuntimeError("Missing dependency: reportlab. Run: pip install reportlab") from e

        self._ensure_output_dir()

        # Keep only file name (not full source path) in the header.
        file_display_name = os.path.basename(source_file) if source_file else "Live Recording"

        duration_str = self._format_duration(duration_seconds, start_time, end_time)

        summary_paragraphs = self.formatter.split_summary_into_paragraphs(summary or "")

        pdf_path = self._build_pdf_path(source_file)

        styles = getSampleStyleSheet()
        report_styles = ReportStyleFactory.build(styles, colors, ParagraphStyle, TA_JUSTIFY)

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

        story.append(Paragraph("Executive Overview", report_styles["heading"]))
        for paragraph in summary_paragraphs:
            story.append(Paragraph(escape(paragraph), report_styles["body"]))
        story.append(Spacer(1, 0.1 * inch))

        # Topics Discussed section
        if topics:
            story.append(Paragraph("Topics Discussed", report_styles["heading"]))
            topic_list = topics if isinstance(topics, list) else [topics]
            for idx, topic in enumerate(topic_list, 1):
                clean_topic = str(topic).strip()
                if clean_topic:
                    story.append(Paragraph(f"<b>{idx}.</b> {escape(clean_topic)}", report_styles["body"]))
            story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("Action Items", report_styles["heading"]))
        items = self.formatter.extract_llama_action_items(summary or "")
        if items:
            for item in items:
                clean_item = item.strip()
                if not clean_item:
                    continue
                story.append(Paragraph(f"<b>• Action Item:</b> {escape(clean_item)}", report_styles["action_item"]))
                story.append(
                    Paragraph(
                        f"<i>Task:</i> {escape(self.formatter.build_action_explanation(clean_item))}",
                        report_styles["action_help"],
                    )
                )
        else:
            story.append(Paragraph("<i>No action items were identified in the generated summary.</i>", report_styles["body"]))

        # Optional section for future use:
        story.append(Spacer(1, 0.14 * inch))
        story.append(Paragraph("Full Transcription", report_styles["heading"]))
        safe_content = escape(content or "[Empty Transcript]").replace("\n", "<br/>")
        story.append(Paragraph(safe_content, report_styles["transcript"]))

        doc.build(story)
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

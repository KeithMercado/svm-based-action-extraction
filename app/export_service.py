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
        item = (action_item or "").lower()
        if any(keyword in item for keyword in ["review", "annotate", "source", "citation", "credible", "wiki", "blog"]):
            return (
                "Gather all references, verify each source against academic journals or official publications, "
                "replace weak references, and submit a finalized citation list for review."
            )
        if any(keyword in item for keyword in ["schedule", "meeting", "sync", "follow-up"]):
            return (
                "Assign an owner, lock the schedule, share agenda points ahead of time, and document final decisions "
                "after the session."
            )
        if any(keyword in item for keyword in ["deadline", "submit", "deliver"]):
            return (
                "Break work into checkpoints, assign accountable members, track progress in weekly updates, and escalate "
                "blockers before the deadline."
            )
        if any(keyword in item for keyword in ["test", "validate", "check", "qa"]):
            return (
                "Define acceptance criteria, run validation scenarios, log defects clearly, and close the item only after "
                "successful retesting."
            )
        return (
            "Define the expected output, assign a single owner, set a due date, and post progress updates in the team "
            "tracker until closure."
        )


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

    def generate_pdf(
        self,
        content,
        action_items=None,
        summary=None,
        duration_seconds=None,
        start_time=None,
        end_time=None,
        source_file=None,
    ):
        """Generate a professional PDF report with a clean narrative summary and SVM action items."""
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

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        pdf_path = os.path.join(self.output_dir, f"Meeting_Minutes_{timestamp}.pdf")

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

        story.append(Paragraph("Action Items from SVM Model", report_styles["heading"]))
        items = action_items or []
        if items:
            for item in items:
                clean_item = item.strip()
                if not clean_item:
                    continue
                story.append(Paragraph(f"<b>• Action Item:</b> {escape(clean_item)}", report_styles["action_item"]))
                story.append(
                    Paragraph(
                        f"<i>How to do this:</i> {escape(self.formatter.build_action_explanation(clean_item))}",
                        report_styles["action_help"],
                    )
                )
        else:
            story.append(Paragraph("<i>No specific action items were detected by the SVM model.</i>", report_styles["body"]))

        # Optional section for future use:
        # story.append(Spacer(1, 0.14 * inch))
        # story.append(Paragraph("Full Transcription", report_styles["heading"]))
        # safe_content = escape(content or "[Empty Transcript]").replace("\n", "<br/>")
        # story.append(Paragraph(safe_content, report_styles["transcript"]))

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

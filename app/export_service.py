import os
from datetime import datetime


class ExportService:
    def __init__(self):
        self.output_dir = os.path.join("output", "pdf")

    def _ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_pdf(
        self,
        content,
        action_items=None,
        summary=None,
        segment_summaries=None,
        source_file=None,
    ):
        """Generate a PDF report that includes transcript, action items, and summary."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        except ModuleNotFoundError as e:
            raise RuntimeError("Missing dependency: reportlab. Run: pip install reportlab") from e

        self._ensure_output_dir()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_path = os.path.join(self.output_dir, f"meeting_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Meeting Processing Report", styles["Title"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        if source_file:
            story.append(Paragraph(f"Source File: {source_file}", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(Paragraph((summary or "No summary generated.").replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Action Items", styles["Heading2"]))
        items = action_items or []
        if items:
            for item in items:
                story.append(Paragraph(f"- {item}", styles["Normal"]))
        else:
            story.append(Paragraph("- None identified", styles["Normal"]))
        story.append(Spacer(1, 12))

        if segment_summaries:
            story.append(Paragraph("Segment Summaries", styles["Heading2"]))
            for idx, seg_summary in enumerate(segment_summaries, start=1):
                story.append(Paragraph(f"Segment {idx}: {seg_summary}", styles["Normal"]))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 12))

        story.append(Paragraph("Full Transcript", styles["Heading2"]))
        story.append(Paragraph((content or "").replace("\n", "<br/>"), styles["Normal"]))

        doc.build(story)
        return pdf_path
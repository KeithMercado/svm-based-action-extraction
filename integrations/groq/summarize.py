import os


def summarize_with_groq(text: str, action_items: list[str] | None = None, model: str | None = None) -> str:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: python-dotenv. Run: pip install python-dotenv") from e

    try:
        from groq import Groq
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: groq. Run: pip install groq") from e

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

    chosen_model = model or os.getenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant")

    actions = action_items or []
    actions_block = "\n".join([f"- {a}" for a in actions[:30]])

    system_prompt = (
        "You are an assistant that summarizes meeting transcripts. "
        "Produce concise, readable English output."
    )

    user_prompt = (
        "Summarize the meeting content below in 3-6 sentences. "
        "Then add a section named 'Action Items' with bullet points. "
        "If no clear action items exist, write '- None identified'.\n\n"
        f"Transcript:\n{text[:12000]}\n\n"
        f"Detected Action Candidates:\n{actions_block if actions_block else '- None provided'}"
    )

    client = Groq(api_key=api_key)
    result = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )

    content = result.choices[0].message.content if result.choices else ""
    if not content:
        raise RuntimeError("No summary text returned by Groq.")

    return content.strip()

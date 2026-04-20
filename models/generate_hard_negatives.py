import csv
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic hard-negative information items.")
    parser.add_argument("--rows", type=int, default=20000, help="Number of rows to generate")
    parser.add_argument(
        "--output",
        type=str,
        default="data/hard_negative_information_items_20000.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    random.seed(42)

    target_rows = max(1000, int(args.rows))
    out_path = Path(args.output)

    subjects = [
        "the team",
        "the group",
        "the client",
        "finance",
        "operations",
        "engineering",
        "product",
        "marketing",
        "support",
        "qa",
        "leadership",
        "the committee",
    ]
    artifacts = [
        "timeline",
        "proposal",
        "draft",
        "status",
        "report",
        "dashboard",
        "plan",
        "scope",
        "budget",
        "risk register",
        "summary",
        "meeting notes",
        "agenda",
    ]
    states = [
        "is unchanged",
        "is still under review",
        "was discussed",
        "was reviewed",
        "was presented",
        "remains pending",
        "remains open",
        "is being monitored",
        "is already circulated",
        "is for reference",
        "is part of the discussion",
    ]
    time_markers = [
        "today",
        "earlier",
        "this week",
        "this month",
        "in the call",
        "during the meeting",
        "in the discussion",
        "in the last update",
        "in the sync",
    ]
    closures = [
        "no final decision yet",
        "no further updates",
        "for context only",
        "for information only",
        "no action required",
        "discussion-only",
        "for tracking purposes",
        "for visibility",
    ]
    acknowledgements = [
        "thank you",
        "thanks everyone",
        "noted",
        "got it",
        "copy that",
        "understood",
        "acknowledged",
        "message received",
        "thanks for the update",
        "appreciate it",
    ]
    openers = [
        "good morning everyone",
        "good afternoon everyone",
        "good evening everyone",
        "welcome everyone",
        "just sharing an update",
        "for your information",
        "fyi",
    ]
    transitions = [
        "moving to the next topic",
        "let's proceed to the next agenda item",
        "closing this discussion",
        "that is all from my side",
        "no additional comments",
        "the meeting is adjourned",
    ]

    qualifiers = [
        "for context",
        "for visibility",
        "for documentation",
        "for reporting",
        "for reference",
        "for alignment",
        "for the record",
        "for awareness",
        "for meeting notes",
        "for tracking",
    ]

    channels = [
        "in this meeting",
        "in the team sync",
        "in the status call",
        "during the review call",
        "during the checkpoint",
        "in the project discussion",
        "in today's update",
        "in the planning call",
        "in the weekly review",
        "in the follow-on discussion",
    ]

    hedges = [
        "as discussed",
        "as mentioned",
        "as noted",
        "as shared",
        "as reviewed",
        "as clarified",
        "as observed",
        "as highlighted",
        "as presented",
        "as captured",
    ]

    endings = [
        "no decision was finalized",
        "no ownership was assigned",
        "no action item was created",
        "no commitment was requested",
        "no due date was confirmed",
        "no follow-up was required",
        "this remains informational",
        "this is discussion-only",
        "this is status-only",
        "this remains an update",
    ]

    blocked = [
        "please send",
        "need to",
        "must",
        "assign",
        "deadline",
        "follow up",
        "can you",
        "could you",
    ]

    rows = set((item, "information_item") for item in (acknowledgements + openers + transitions))

    for s in subjects:
        for a in artifacts:
            for st in states:
                for ch in channels:
                    for q in qualifiers:
                        text = f"{hedges[(hash(s + a + st + ch + q) % len(hedges))]} {s} said the {a} {st} {ch}, {q}"
                        text = " ".join(text.split()).strip(" .;,").lower()
                        if any(token in text for token in blocked):
                            continue
                        rows.add((text, "information_item"))

    for s in subjects:
        for a in artifacts:
            for tm in time_markers:
                for cl in closures:
                    for end in endings:
                        text = f"{s} discussed the {a} {tm}; {cl}; {end}"
                        text = " ".join(text.split()).strip(" .;,").lower()
                        if any(token in text for token in blocked):
                            continue
                        rows.add((text, "information_item"))

    all_rows = sorted(rows)
    if len(all_rows) < target_rows:
        raise RuntimeError(f"Only generated {len(all_rows)} unique rows; need at least {target_rows}.")

    random.shuffle(all_rows)
    sorted_rows = all_rows[:target_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence", "label"])
        writer.writerows(sorted_rows)

    print(f"Wrote {len(sorted_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

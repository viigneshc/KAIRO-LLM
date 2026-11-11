
import json
from collections import Counter
from pathlib import Path

DATASET = Path("data_v4/kairo_bilingual_v4.jsonl")


def main():
    if not DATASET.exists():
        print("ERROR: Dataset not found:", DATASET)
        return

    total = 0
    invalid = 0
    missing_system = 0
    missing_question = 0
    missing_answer = 0
    duplicate_questions = 0
    tanglish_examples = 0
    english_examples = 0
    seen_questions = set()
    samples = []

    with DATASET.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                invalid += 1
                print(f"Invalid JSON on line {line_number}: {error}")
                continue

            messages = record.get("messages")

            if not isinstance(messages, list):
                invalid += 1
                continue

            system_messages = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") == "system"
            ]
            user_messages = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            assistant_messages = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]

            if not system_messages:
                missing_system += 1

            if not user_messages or not str(
                user_messages[-1].get("content", "")
            ).strip():
                missing_question += 1
                continue

            if not assistant_messages or not str(
                assistant_messages[-1].get("content", "")
            ).strip():
                missing_answer += 1
                continue

            question = str(user_messages[-1]["content"]).strip()
            answer = str(assistant_messages[-1]["content"]).strip()

            normalized = " ".join(question.lower().split())

            if normalized in seen_questions:
                duplicate_questions += 1
            else:
                seen_questions.add(normalized)

            # Heuristic only: this estimates language style using
            # common Tanglish words. It is not a perfect classifier.
            tanglish_words = [
                "na", "enna", "pannu", "panna", "epdi",
                "eppadi", "irukku", "panradhu", "kudunga",
                "sollu", "moolama", "thevai", "aagum",
                "kathukkum", "puriyum", "mudiyum",
            ]

            q_lower = question.lower()
            if any(
                word in q_lower.split()
                for word in tanglish_words
            ):
                tanglish_examples += 1
            else:
                english_examples += 1

            if len(samples) < 10:
                samples.append((question, answer))

    print("=" * 55)
    print("KAIRO V4 DATASET AUDIT")
    print("=" * 55)
    print("File:", DATASET)
    print("Total examples:", total)
    print("Invalid structures/JSON:", invalid)
    print("Missing system prompts:", missing_system)
    print("Missing questions:", missing_question)
    print("Missing answers:", missing_answer)
    print("Duplicate normalized questions:", duplicate_questions)
    print("Likely Tanglish questions:", tanglish_examples)
    print("Other questions:", english_examples)

    print("\nSample examples:")
    for index, (question, answer) in enumerate(samples, start=1):
        print(f"\n[{index}]")
        print("Q:", question)
        print("A:", answer)

    print("\nAudit completed. Dataset was not modified.")


if __name__ == "__main__":
    main()

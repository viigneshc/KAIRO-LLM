
import json
from pathlib import Path

DATASET = Path("data_v2/kairo_bilingual_v2.jsonl")

def main():
    if not DATASET.exists():
        print("Dataset not found:", DATASET)
        return

    with DATASET.open("r", encoding="utf-8-sig") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("=" * 55)
    print("KAIRO DATASET AUDIT")
    print("=" * 55)
    print("Total examples:", len(rows))

    invalid = 0
    missing_system = 0
    missing_user = 0
    missing_answer = 0
    duplicates = set()
    duplicate_count = 0

    for index, row in enumerate(rows, start=1):
        messages = row.get("messages", [])

        if len(messages) != 3:
            invalid += 1
            print(f"Invalid message structure: example {index}")
            continue

        system = messages[0].get("content", "").strip()
        question = messages[1].get("content", "").strip()
        answer = messages[2].get("content", "").strip()

        if not system:
            missing_system += 1
        if not question:
            missing_user += 1
        if not answer:
            missing_answer += 1

        key = (question.casefold(), answer.casefold())
        if key in duplicates:
            duplicate_count += 1
        duplicates.add(key)

    print("Invalid structures:", invalid)
    print("Missing system prompts:", missing_system)
    print("Missing questions:", missing_user)
    print("Missing answers:", missing_answer)
    print("Duplicate question-answer pairs:", duplicate_count)

    print("\nSample examples:")
    for index, row in enumerate(rows[:8], start=1):
        messages = row.get("messages", [])
        if len(messages) == 3:
            print(f"\n[{index}]")
            print("Q:", messages[1]["content"])
            print("A:", messages[2]["content"])

    print("\nAudit finished. Dataset was not modified.")

if __name__ == "__main__":
    main()


import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data_v7" / "kairo_bilingual_v7.jsonl"
OUTPUT_FILE = ROOT / "data_v7" / "kairo_bilingual_v7_clean.jsonl"

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful and accurate AI assistant. "
    "Answer English questions in clear English. "
    "For Tanglish requests, use natural Tamil written only in English letters. "
    "Never use Tamil script in Tanglish answers. "
    "Answer directly, avoid repetition, and follow the requested format. "
    "For coding requests, provide complete, runnable programs when requested. "
    "Never invent facts."
)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Existing file was not overwritten."
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if "messages" not in record:
                raise ValueError(f"Missing messages at line {line_number}")

            messages = record["messages"]
            system_messages = [
                message for message in messages
                if message.get("role") == "system"
            ]

            if system_messages:
                system_messages[0]["content"] = SYSTEM_PROMPT
                messages[:] = [
                    message for message in messages
                    if message.get("role") != "system"
                    or message is system_messages[0]
                ]
            else:
                messages.insert(
                    0,
                    {"role": "system", "content": SYSTEM_PROMPT}
                )

            record["messages"] = messages
            records.append(record)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("KAIRO V7 dataset cleaning complete")
    print("----------------------------------")
    print("Records:", len(records))
    print("Unique system prompt: 1")
    print("Output:", OUTPUT_FILE)
    print("Original V7 dataset was not modified.")

if __name__ == "__main__":
    main()

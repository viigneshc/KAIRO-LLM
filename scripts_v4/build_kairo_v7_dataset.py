
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = ROOT / "data_v6" / "kairo_bilingual_v6.jsonl"
OUTPUT_FILE = ROOT / "data_v7" / "kairo_bilingual_v7.jsonl"

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful AI assistant. "
    "For Tanglish requests, write Tamil using English letters only. "
    "Do not use Tamil script in Tanglish answers. "
    "Use natural, simple, conversational Tanglish. "
    "Answer directly, avoid repetition, and follow the requested format. "
    "For coding requests, provide complete, runnable programs when requested."
)

NEW_EXAMPLES = [
    {
        "user": "Machine Learning na enna? Natural Tanglish la 4 simple sentences la explain pannu. Tamil script use panna vendam.",
        "assistant": (
            "Machine Learning na computer-ku examples use panni learn panna solli kudukkura oru AI technique. "
            "Idhula computer data-la irukkura patterns-ai identify pannum. "
            "Andha patterns use panni pudhu data-ku prediction pannum. "
            "Example-ku, neraya spam emails-ai analyze panni, pudhu spam emails-ai identify pannum."
        ),
    },
    {
        "user": "Cloud computing na enna? Oru real-life example-oda simple Tanglish la explain pannu. Tamil script vendaam.",
        "assistant": (
            "Cloud computing na internet moolama storage, servers, software maadhiri services use panradhu. "
            "Example-ku, phone-la Google Drive use panni photos save pannina, adhu cloud storage. "
            "Photos phone-la mattum illaama online server-la store aagum. "
            "Internet irundha vera device-layum andha photos-ai access pannalaam."
        ),
    },
    {
        "user": "DBSCAN clustering na enna? Core point and noise point-ai simple Tanglish la explain pannu.",
        "assistant": (
            "DBSCAN na data points nerukkama irukkura density-ai base panni clusters create pannura algorithm. "
            "Core point-na adha suthi required number of nearby points irukkura point. "
            "Noise point-na endha cluster-layum seradha point. "
            "Example-ku, nerukkama irukkura location points oru cluster aagum; thaniya irukkura point noise-a irukkalaam."
        ),
    },
    {
        "user": "Python-la user input vaangi even or odd print pannura complete program kudu. Code mattum.",
        "assistant": (
            "num = int(input('Enter an integer: '))\n\n"
            "if num % 2 == 0:\n"
            "    print('Even')\n"
            "else:\n"
            "    print('Odd')"
        ),
    },
    {
        "user": "Python-la user input use panni factorial calculate pannura complete program kudu. Negative input reject pannanum; 0 factorial 1. Code mattum.",
        "assistant": (
            "num = int(input('Enter a non-negative integer: '))\n\n"
            "if num < 0:\n"
            "    print('Invalid input')\n"
            "else:\n"
            "    fact = 1\n"
            "    for i in range(1, num + 1):\n"
            "        fact *= i\n"
            "    print(fact)"
        ),
    },
    {
        "user": "Python-la user input vaangi number square print pannura complete program kudu. Code mattum.",
        "assistant": (
            "num = float(input('Enter a number: '))\n"
            "print(num * num)"
        ),
    },
    {
        "user": "Python-la 1 lendhu 10 vara numbers print pannura code kudu. Code mattum.",
        "assistant": (
            "for i in range(1, 11):\n"
            "    print(i)"
        ),
    },
    {
        "user": "Python-la list-la irukkura numbers total calculate pannura complete program kudu. Code mattum.",
        "assistant": (
            "numbers = [10, 20, 30, 40]\n"
            "total = sum(numbers)\n"
            "print(total)"
        ),
    },
    {
        "user": "Overfitting na enna? Simple Tanglish la 3 sentences-la explain pannu. Tamil script use panna vendaam.",
        "assistant": (
            "Overfitting na machine learning model training data-ai romba nalla memorize pannidradhu. "
            "Aana pudhu data varumbodhu correct-aa predict panna struggle pannalaam. "
            "Idhai reduce panna more diverse data, regularization, and validation use pannalaam."
        ),
    },
    {
        "user": "Python-la list-la largest number find pannura complete program kudu. Code mattum.",
        "assistant": (
            "numbers = [12, 45, 7, 89, 23]\n\n"
            "if numbers:\n"
            "    print(max(numbers))\n"
            "else:\n"
            "    print('List is empty')"
        ),
    },
    {
        "user": "API na enna? Simple Tanglish la oru example-oda explain pannu. Tamil script vendaam.",
        "assistant": (
            "API na rendu software applications communicate panna help pannura interface. "
            "Example-ku, weather app API use panni weather server-kitta temperature details kekkum. "
            "Server response anuppum; app adhai user-kku display pannum."
        ),
    },
    {
        "user": "SQL injection na enna? Simple Tanglish la explain pannu.",
        "assistant": (
            "SQL injection na unsafe user input moolama database query-ai manipulate panna try panra attack. "
            "Input-ai direct-aa SQL query-oda join pannina risk adhigam. "
            "Parameterized queries or prepared statements use pannina indha risk-ai reduce pannalaam."
        ),
    },
]


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_messages(record, line_number):
    messages = record.get("messages")

    if not isinstance(messages, list):
        raise ValueError(
            f"Line {line_number}: 'messages' must be a list."
        )

    user_messages = [
        item.get("content", "")
        for item in messages
        if item.get("role") == "user"
    ]

    assistant_messages = [
        item.get("content", "")
        for item in messages
        if item.get("role") == "assistant"
    ]

    if not user_messages or not assistant_messages:
        raise ValueError(
            f"Line {line_number}: user/assistant message missing."
        )

    question = user_messages[-1]
    answer = assistant_messages[-1]

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"Line {line_number}: invalid user question."
        )

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"Line {line_number}: invalid assistant answer."
        )

    return question, answer


def create_record(question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"V6 dataset not found: {SOURCE_FILE}"
        )

    records = []
    seen_questions = set()

    with SOURCE_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            question, _ = extract_messages(record, line_number)
            key = normalize(question)

            if key not in seen_questions:
                seen_questions.add(key)
                records.append(record)

    source_count = len(records)
    added_count = 0

    for example in NEW_EXAMPLES:
        question = example["user"]
        answer = example["assistant"]
        key = normalize(question)

        if key in seen_questions:
            continue

        records.append(create_record(question, answer))
        seen_questions.add(key)
        added_count += 1

    # Refuse to overwrite an existing V7 dataset.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"V7 dataset already exists: {OUTPUT_FILE}\n"
            "Existing file was not changed. Rename or back it up "
            "before creating another version."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    validated_count = 0

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            extract_messages(record, line_number)
            validated_count += 1

    print("\nKAIRO V7 dataset build complete")
    print("--------------------------------")
    print(f"Unique V6 source records: {source_count}")
    print(f"New examples added:      {added_count}")
    print(f"Total V7 records:         {len(records)}")
    print(f"Records validated:        {validated_count}")
    print(f"Output: {OUTPUT_FILE}")
    print("Existing V1-V6 datasets and adapters were not modified.")


if __name__ == "__main__":
    main()

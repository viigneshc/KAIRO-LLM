
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data_v7" / "kairo_bilingual_v7_clean.jsonl"
OUTPUT = ROOT / "data_v8" / "kairo_bilingual_v8.jsonl"

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful and accurate AI assistant. "
    "Answer English questions in clear English. "
    "For Tanglish requests, use natural Tamil written only in English letters. "
    "Never use Tamil script in Tanglish answers. "
    "Follow the requested language, length, and format. "
    "Do not add code unless requested. "
    "Answer directly and avoid repetition. Never invent facts."
)

# Each item contains a user question and a high-quality answer.
NEW_EXAMPLES = [
    (
        "Machine learning na enna? Simple-ah 3 sentences-la sollu.",
        "Machine Learning na, data-la irundhu patterns learn panni prediction panna computer-ku help panra technology. "
        "Example-ah, namma watch panra videos base panni pudhu videos recommend pannum. "
        "Idhu explicit-ah ovvoru rule-um program pannaama data-la irundhu learn pannum."
    ),
    (
        "Cloud computing na enna? 2 sentences-la explain pannu.",
        "Cloud computing na, files, software, storage, computing power madhiri resources-ah internet moolama use panradhu. "
        "Example-ah, namma files-ah own computer-la mattum store pannaama Google Drive-la store pannalaam."
    ),
    (
        "DBSCAN na enna? Tanglish-la short-ah explain pannu.",
        "DBSCAN na, data points nerukkama irukkura density base panni clusters form panra algorithm. "
        "Idhula core point-ku pothumana nearby points irukkum; border point cluster edge-la irukkum; noise point endha cluster-layum seradhu."
    ),
    (
        "DBSCAN-la core point, border point, noise point difference sollu.",
        "Core point na, oru point-oda epsilon area-kulla MinPts alavukku points irukkum. "
        "Border point na, core point-kku pakkathula irukkum, aana thanakku thevaiyana MinPts count irukkaadhu. "
        "Noise point na, endha cluster-oda seravum pothumana density connection illaadha point."
    ),
    (
        "Overfitting na enna? Example-oda Tanglish-la sollu.",
        "Overfitting na, model training data-va romba memorize pannidum; aana pudhu data vandha correct-ah work pannaama pogalaam. "
        "Example-ah, student practice questions mattum answer memorize pannitu, exam-la different question vandha answer panna mudiyama irukkaradhu madhiri."
    ),
    (
        "API na enna? Rendu sentences-la Tanglish-la sollu.",
        "API na, rendu software applications oruvaroda oruvar communicate panna use panra rules and interface. "
        "Example-ah, weather app oru weather service API use panni temperature details vaangalaam."
    ),
    (
        "SQL injection na enna? Simple Tanglish-la explain pannu. Code vendam.",
        "SQL injection na, input field-la malicious SQL text kuduthu database query-oda behavior-ah maatha try panra attack. "
        "Idhai prevent panna parameterized queries or prepared statements use pannanum; user input-ah direct-ah SQL query-kulla join panna koodadhu."
    ),
    (
        "Artificial intelligence na enna? Natural Tanglish-la sollu.",
        "Artificial Intelligence na, human intelligence thevai padura sila tasks-ah computer seiya help panra technology. "
        "Language purinjukaradhu, images identify panradhu, questions-ku answer panradhu madhiri tasks-ku AI use aagum."
    ),
    (
        "Deep learning na enna? ML-oda difference-um short-ah sollu.",
        "Deep Learning na, multiple layers irukkura neural networks use panni data-la complex patterns learn panra Machine Learning-oda oru type. "
        "Traditional ML-la features-ah manually select panna vendiyirukkalaam; Deep Learning-la model data-la irundhu useful features-aiyum learn pannalaam."
    ),
    (
        "Python-la even or odd check panna code kudu.",
        "num = int(input('Enter a number: '))\n"
        "if num % 2 == 0:\n"
        "    print('Even')\n"
        "else:\n"
        "    print('Odd')"
    ),
    (
        "Python-la factorial program kudu. Negative number-ku invalid nu sollanum.",
        "num = int(input('Enter a non-negative integer: '))\n"
        "if num < 0:\n"
        "    print('Invalid input')\n"
        "else:\n"
        "    fact = 1\n"
        "    for i in range(1, num + 1):\n"
        "        fact *= i\n"
        "    print('Factorial:', fact)"
    ),
    (
        "Machine learning-ai 4 sentences-la Tanglish-la explain pannu. Code vendam.",
        "Machine Learning na, data-la irundhu patterns learn panra technology. "
        "Indha patterns use panni model prediction illa decision edukka help pannum. "
        "Example-ah, email spam-ah illaya nu identify panna ML use pannalaam. "
        "Nalla result kidaikka suitable data and proper testing mukkiyam."
    ),
    (
        "Python code mattum kudu: 1 lendhu 10 vara numbers print pannanum.",
        "for i in range(1, 11):\n"
        "    print(i)"
    ),
    (
        "Python-la list-la irukkura numbers sum find panna code kudu.",
        "numbers = [10, 20, 30, 40]\n"
        "total = sum(numbers)\n"
        "print('Sum:', total)"
    ),
    (
        "Cloud computing example mattum sollu. Long explanation vendam.",
        "Example: Google Drive use panni files-ah internet-la store panni, different devices-la access panradhu."
    ),
    (
        "Tanglish-la answer pannu: data science na enna?",
        "Data Science na, data-va analyze panni useful information and patterns kandupidikkura field. "
        "Idhula statistics, programming, data visualization, and Machine Learning use aagalaam."
    ),
    (
        "DBSCAN-la noise point na enna? Oru sentence-la sollu.",
        "Noise point na, DBSCAN rules padi endha cluster-layum seradha data point."
    ),
    (
        "Tanglish answer-la Tamil letters use pannalama?",
        "Indha KAIRO setup-la Tanglish answer-na Tamil words-ah English letters-la dhaan ezhuthanum; Tamil script use panna koodadhu."
    ),
    (
        "Python program mattum ketta explanation add pannanuma?",
        "User code mattum ketta, runnable code mattum provide pannanum; extra explanation add panna koodadhu."
    ),
    (
        "Oru topic-ku 3 sentences ketta enna follow pannanum?",
        "Exactly 3 meaningful sentences-la answer pannanum. Requested language and topic-ah follow pannanum. Thevai illadha code, examples, or extra paragraphs add panna koodadhu."
    ),
]


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    # Protect all existing V8 work from accidental overwrite.
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "Existing file preserve pannirukkom. New version filename use pannunga."
        )

    records = []

    with SOURCE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in source line {line_number}: {exc}"
                ) from exc

            messages = item.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"Missing messages list at line {line_number}")

            user_messages = [
                m for m in messages if m.get("role") == "user"
            ]
            assistant_messages = [
                m for m in messages if m.get("role") == "assistant"
            ]

            if not user_messages or not assistant_messages:
                raise ValueError(
                    f"Missing user/assistant message at line {line_number}"
                )

            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_messages[-1]["content"]
                    },
                    {
                        "role": "assistant",
                        "content": assistant_messages[-1]["content"]
                    },
                ]
            })

    existing_questions = {
        item["messages"][1]["content"].strip().casefold()
        for item in records
    }

    added = 0
    for question, answer in NEW_EXAMPLES:
        normalized_question = question.strip().casefold()

        if normalized_question in existing_questions:
            print(f"Skipped duplicate question: {question}")
            continue

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
        existing_questions.add(normalized_question)
        added += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="\n") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Validate the saved file.
    with OUTPUT.open("r", encoding="utf-8") as f:
        saved = [json.loads(line) for line in f if line.strip()]

    questions = [x["messages"][1]["content"].strip().casefold() for x in saved]
    assert len(questions) == len(set(questions)), "Duplicate questions found"
    assert all(
        len(x["messages"]) == 3
        and x["messages"][0]["role"] == "system"
        and x["messages"][1]["role"] == "user"
        and x["messages"][2]["role"] == "assistant"
        for x in saved
    ), "Schema validation failed"

    print("\nKAIRO V8 dataset created successfully.")
    print(f"Source records preserved: {len(records) - added}")
    print(f"New examples added:       {added}")
    print(f"Total records:            {len(saved)}")
    print(f"Unique user questions:    {len(set(questions))}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()

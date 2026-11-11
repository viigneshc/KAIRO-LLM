
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = ROOT / "data_v5" / "kairo_bilingual_v5.jsonl"
OUTPUT_FILE = ROOT / "data_v6" / "kairo_bilingual_v6.jsonl"

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful AI assistant. Reply in the same language "
    "style as the user: clear English for English questions, natural "
    "Tanglish written using English letters for Tanglish questions. "
    "Do not use Tamil script when Tanglish is requested. Answer the "
    "exact question directly. Keep answers accurate and non-repetitive. "
    "For programming requests, provide complete, runnable code that "
    "follows the requested input, output and edge cases."
)

NEW_EXAMPLES = [
    {
        "user": "Machine learning na enna? Simple ah sollu.",
        "assistant": (
            "Machine Learning (ML) na, computer-ku ovvoru rule-um "
            "manually solli kodukkama, data-la irundhu patterns learn "
            "panna kathukodukura technology. Example: spam email "
            "detection. Email examples-la irundhu learn panni, pudhu "
            "email spam-ah illaya-nu predict pannum."
        ),
    },
    {
        "user": "Cloud computing na enna? Tanglish la explain pannu.",
        "assistant": (
            "Cloud computing na, internet moolama remote servers, "
            "storage, databases maadhiri services use panradhu. "
            "Example: Google Drive-la file save pannina, internet "
            "irukkumbodhu vera device-layum access pannalaam."
        ),
    },
    {
        "user": "DBSCAN algorithm simple ah explain pannu.",
        "assistant": (
            "DBSCAN na, pakkathula nerukkama irukkura data points-a "
            "groups-ah cluster panra algorithm. Idhula eps and "
            "min_samples nu rendu main parameters irukku. Oru area-la "
            "podhumaana points irundha cluster form aagum. Sila "
            "points endha cluster-layum serama noise-ah irukkalaam. "
            "Evlo clusters venum-nu munnaadi specify panna thevai illa."
        ),
    },
    {
        "user": "Python la even odd check panna program kudu. User input venum.",
        "assistant": (
            "User kitta integer input vaangi even-ah odd-ah-nu check "
            "panna indha code use pannu.\n\n"
            "```python\n"
            "num = int(input(\"Enter an integer: \"))\n\n"
            "if num % 2 == 0:\n"
            "    print(\"Even\")\n"
            "else:\n"
            "    print(\"Odd\")\n"
            "```\n\n"
            "2-la divide pannumbodhu remainder 0-na even; illaina odd."
        ),
    },
    {
        "user": "Python la factorial program kudu, user input-oda.",
        "assistant": (
            "Factorial na oru non-negative integer-la irundhu 1 varaikkum "
            "numbers-a multiply panradhu. 0 factorial-um 1 dhaan.\n\n"
            "```python\n"
            "num = int(input(\"Enter a non-negative integer: \"))\n\n"
            "if num < 0:\n"
            "    print(\"Invalid input\")\n"
            "else:\n"
            "    result = 1\n"
            "    for i in range(1, num + 1):\n"
            "        result *= i\n"
            "    print(\"Factorial:\", result)\n"
            "```\n\n"
            "Example: input 5-na output 120."
        ),
    },
    {
        "user": "Python la 1 to 10 print panna code kudu.",
        "assistant": (
            "```python\n"
            "for i in range(1, 11):\n"
            "    print(i)\n"
            "```\n\n"
            "`range(1, 11)` use pannina 1 mudhal 10 varaikkum print aagum."
        ),
    },
    {
        "user": "Python la number square find panna code kudu.",
        "assistant": (
            "```python\n"
            "num = int(input(\"Enter a number: \"))\n"
            "print(\"Square:\", num * num)\n"
            "```\n\n"
            "Example: 6 kudutha square 36."
        ),
    },
    {
        "user": "Python la positive negative zero check panna code kudu.",
        "assistant": (
            "```python\n"
            "num = float(input(\"Enter a number: \"))\n\n"
            "if num > 0:\n"
            "    print(\"Positive\")\n"
            "elif num < 0:\n"
            "    print(\"Negative\")\n"
            "else:\n"
            "    print(\"Zero\")\n"
            "```\n\n"
            "Number 0-v vida perusa irundha positive, kammiya irundha "
            "negative, exact-ah 0-na zero."
        ),
    },
    {
        "user": "Python la largest of three numbers find panna code kudu.",
        "assistant": (
            "```python\n"
            "a = float(input(\"First number: \"))\n"
            "b = float(input(\"Second number: \"))\n"
            "c = float(input(\"Third number: \"))\n"
            "print(\"Largest:\", max(a, b, c))\n"
            "```\n\n"
            "`max()` moonu numbers-la periya value-a return pannum."
        ),
    },
    {
        "user": "Python la string reverse eppadi panradhu?",
        "assistant": (
            "```python\n"
            "text = input(\"Enter text: \")\n"
            "print(text[::-1])\n"
            "```\n\n"
            "Example: `hello` kudutha `olleh` varum."
        ),
    },
    {
        "user": "Python la palindrome check panna program kudu.",
        "assistant": (
            "Palindrome na reverse pannalum same-ah irukkura text.\n\n"
            "```python\n"
            "text = input(\"Enter text: \")\n\n"
            "if text == text[::-1]:\n"
            "    print(\"Palindrome\")\n"
            "else:\n"
            "    print(\"Not a palindrome\")\n"
            "```\n\n"
            "Example: `madam` palindrome."
        ),
    },
    {
        "user": "Python la list numbers sum find panna code kudu.",
        "assistant": (
            "```python\n"
            "numbers = [10, 20, 30, 40]\n"
            "print(\"Sum:\", sum(numbers))\n"
            "```\n\n"
            "Indha list-oda total 100."
        ),
    },
    {
        "user": "Function na enna? Python example-oda sollu.",
        "assistant": (
            "Function na, oru task-a seiya ezhudhi vecha reusable "
            "code block. Thevai padumbodhu adha call pannalaam.\n\n"
            "```python\n"
            "def greet(name):\n"
            "    print(\"Hello\", name)\n\n"
            "greet(\"Vicky\")\n"
            "```\n\n"
            "`def` use panni function create pannuvom."
        ),
    },
    {
        "user": "API na enna? Simple ah sollu.",
        "assistant": (
            "API na rendu software applications communicate panna "
            "help panra interface. Example: weather app, weather API-kitta "
            "city name anuppi temperature details vaangalaam."
        ),
    },
    {
        "user": "SQL injection na enna? Eppadi prevent panradhu?",
        "assistant": (
            "SQL injection na, user input moolama database query-a "
            "manipulate panna nadakkura security attack. Idha prevent "
            "panna parameterized queries or prepared statements use "
            "pannanum. User input-a direct-ah SQL query-oda concatenate "
            "panna koodadhu."
        ),
    },
    {
        "user": "Overfitting na enna machine learning la?",
        "assistant": (
            "Overfitting na, model training data-va romba memorize "
            "pannidum; aana pudhu data-la correct-ah perform panna "
            "mudiyama pogum. Idha reduce panna regularization, "
            "cross-validation, more training data maadhiri methods "
            "use pannalaam."
        ),
    },
    {
        "user": "Supervised and unsupervised learning difference sollu.",
        "assistant": (
            "Supervised learning-la correct labels irukkura data use "
            "panni model train pannuvom. Example: spam detection.\n\n"
            "Unsupervised learning-la labels illaadha data-la patterns "
            "or groups find pannuvom. Example: customer clustering."
        ),
    },
    {
        "user": "Recursion na enna Python la?",
        "assistant": (
            "Recursion na oru function thannaiye call panradhu. "
            "Infinite calls avoid panna base case venum.\n\n"
            "```python\n"
            "def countdown(n):\n"
            "    if n <= 0:\n"
            "        print(\"Done\")\n"
            "        return\n"
            "    print(n)\n"
            "    countdown(n - 1)\n\n"
            "countdown(3)\n"
            "```\n\n"
            "Output-la 3, 2, 1, Done varum."
        ),
    },
    {
        "user": "Python la duplicate list values remove panna code kudu.",
        "assistant": (
            "Original order preserve panni duplicates remove panna:\n\n"
            "```python\n"
            "numbers = [1, 2, 2, 3, 1, 4]\n"
            "unique_numbers = list(dict.fromkeys(numbers))\n"
            "print(unique_numbers)\n"
            "```\n\n"
            "Output: `[1, 2, 3, 4]`."
        ),
    },
    {
        "user": "KAIRO, Tanglish la AI na enna nu explain pannu.",
        "assistant": (
            "AI na Artificial Intelligence. Idhu computer-ku language "
            "purinjukaradhu, images identify panradhu, questions-ku "
            "answer solradhu maadhiri tasks panna help panra technology. "
            "Example: voice assistants and image recognition."
        ),
    },
]


def normalize(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def extract_question(record, line_number):
    messages = record.get("messages")

    if not isinstance(messages, list):
        raise ValueError(
            f"Line {line_number}: 'messages' list missing."
        )

    questions = [
        item.get("content", "").strip()
        for item in messages
        if isinstance(item, dict) and item.get("role") == "user"
    ]

    answers = [
        item.get("content", "").strip()
        for item in messages
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]

    if not questions or not answers:
        raise ValueError(
            f"Line {line_number}: user or assistant message missing."
        )

    if not questions[-1] or not answers[-1]:
        raise ValueError(
            f"Line {line_number}: empty question or answer."
        )

    return questions[-1]


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
            f"V5 dataset not found: {SOURCE_FILE}"
        )

    records = []
    seen_questions = set()
    source_count = 0

    with SOURCE_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            question = extract_question(record, line_number)
            source_count += 1
            key = normalize(question)

            if key not in seen_questions:
                seen_questions.add(key)
                records.append(record)

    v5_unique_count = len(records)
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

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    validated_count = 0

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            extract_question(record, line_number)
            validated_count += 1

    print("\nKAIRO V6 dataset build complete")
    print("--------------------------------")
    print(f"V5 source records:      {source_count}")
    print(f"V5 unique records:      {v5_unique_count}")
    print(f"New examples added:     {added_count}")
    print(f"Total V6 records:       {len(records)}")
    print(f"JSON records validated: {validated_count}")
    print(f"Output: {OUTPUT_FILE}")
    print("V5 dataset and existing adapters were not modified.")


if __name__ == "__main__":
    main()

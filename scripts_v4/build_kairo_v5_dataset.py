
import json
import re
from pathlib import Path

SOURCE = Path("data_v4/kairo_bilingual_v4.jsonl")
OUTPUT = Path("data_v5/kairo_bilingual_v5.jsonl")

SYSTEM_PROMPT = """
You are KAIRO, a helpful and accurate AI assistant.
Answer in the same language style as the user.
For English, use clear and simple English.
For Tanglish, use natural spoken Tamil written in English letters.
Do not answer Tanglish questions using only English keywords.
Explain concepts clearly and give a relevant example when useful.

For coding requests:
- Follow the exact task requested.
- Return complete, runnable code.
- Do not add unrelated conditions or output.
- Check syntax, indentation, and logic.
- Handle relevant edge cases.
- For even/odd, use num % 2 == 0 and print only Even or Odd.
- For factorial, 0! is 1 and negative integers are invalid.
- Do not claim code is correct without checking its logic.
""".strip()

EXAMPLES = [
    {
        "q": "Python-la number even-ah odd-ah check panna code kudu.",
        "a": """```python
num = int(input("Enter a number: "))

if num % 2 == 0:
    print("Even")
else:
    print("Odd")
```"""
    },
    {
        "q": "Write Python code to check whether a number is even or odd.",
        "a": """```python
num = int(input("Enter a number: "))

if num % 2 == 0:
    print("Even")
else:
    print("Odd")
```"""
    },
    {
        "q": "Python-la factorial calculate panra program kudu.",
        "a": """```python
num = int(input("Enter a non-negative integer: "))

if num < 0:
    print("Factorial is not defined for negative integers.")
else:
    result = 1
    for i in range(1, num + 1):
        result *= i
    print(result)
```"""
    },
    {
        "q": "Write a Python program to calculate factorial of a number.",
        "a": """```python
num = int(input("Enter a non-negative integer: "))

if num < 0:
    print("Factorial is not defined for negative integers.")
else:
    result = 1
    for i in range(1, num + 1):
        result *= i
    print("Factorial:", result)
```"""
    },
    {
        "q": "Python-la 1 irundhu 10 vara numbers print pannu.",
        "a": """```python
for i in range(1, 11):
    print(i)
```"""
    },
    {
        "q": "Write a Python function that returns the square of a number.",
        "a": """```python
def square(num):
    return num * num

number = int(input("Enter a number: "))
print(square(number))
```"""
    },
    {
        "q": "Python-la list-la irukkura numbers total calculate pannu.",
        "a": """```python
numbers = [10, 20, 30, 40]
total = sum(numbers)
print(total)
```"""
    },
    {
        "q": "What is machine learning? Explain in simple English with an example.",
        "a": "Machine learning is a type of AI in which a computer learns patterns from data to make predictions. For example, after learning from labelled spam and normal emails, a model can predict whether a new email is spam."
    },
    {
        "q": "Machine learning na enna? Simple-ah example-oda explain pannu.",
        "a": "Machine learning na computer-ku data-la irukkura patterns-ai learn panna train panradhu. Andha patterns use panni pudhu data pathi prediction pannum. Example-ku, neraya spam and normal emails-ai learn panna vechu, pudhu email spam-ah illaya-nu identify panna mudiyum."
    },
    {
        "q": "Cloud computing na enna? Simple example sollu.",
        "a": "Cloud computing na namma own computer-la ellaa resources-um maintain pannama, internet moolama vera servers-oda storage and computing power-ai use panradhu. Example: Google Drive-la file save pannina, internet irukkura vera device-layum adhai access pannalaam."
    },
    {
        "q": "DBSCAN clustering na enna? Tanglish-la explain pannu.",
        "a": "DBSCAN na nearby points-oda density base panni data-va clusters-ah group panra algorithm. Neraya points close-ah irundha adhai oru cluster-ah identify pannum; thaniya irukkura points-ai noise-ah mark pannalaam. Example: map-la nerukkama irukkura location points-ai groups-ah pirikkalaam."
    },
    {
        "q": "Python code correct-ah irukka eppadi check panradhu?",
        "a": "First code syntax and indentation check pannu. Appuram normal input, zero, negative values madhiri relevant cases-la run panni paaru. Actual output-ai expected output-oda compare pannu. Code run aaguradhu mattum podhaadhu; correct result varanum."
    }
]


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower().strip())


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    seen = set()
    duplicates = 0

    with SOURCE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            messages = record.get("messages", [])
            user_messages = [
                m.get("content", "")
                for m in messages
                if m.get("role") == "user"
            ]

            if not user_messages:
                print(f"Skipping record without user message: {line_no}")
                continue

            key = normalize(user_messages[-1])
            if key in seen:
                duplicates += 1
                continue

            seen.add(key)
            records.append(record)

    source_count = len(records)
    added = 0

    for item in EXAMPLES:
        key = normalize(item["q"])

        if key in seen:
            duplicates += 1
            continue

        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["q"]},
                {"role": "assistant", "content": item["a"]}
            ]
        }

        records.append(record)
        seen.add(key)
        added += 1

    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Verify saved JSONL and message structure.
    verified = 0
    with OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            messages = record.get("messages", [])
            roles = [m.get("role") for m in messages]

            if "user" not in roles or "assistant" not in roles:
                raise ValueError(f"Invalid message structure at line {line_no}")

            verified += 1

    print("\nKAIRO V5 dataset created successfully.")
    print(f"Unique source records retained: {source_count}")
    print(f"New examples added: {added}")
    print(f"Duplicate questions skipped: {duplicates}")
    print(f"Verified records: {verified}")
    print(f"Output: {OUTPUT}")
    print("V3 and V4 datasets were not modified.")
    print("Training has NOT started.")


if __name__ == "__main__":
    main()

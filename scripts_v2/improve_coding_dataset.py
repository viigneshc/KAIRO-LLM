
import json
from pathlib import Path

SOURCE = Path("data_v2/kairo_bilingual_v2.jsonl")
OUTPUT = Path("data_v2/kairo_bilingual_v3.jsonl")

SYSTEM = """You are KAIRO, a helpful bilingual AI assistant.
Answer in English when the user asks in English.
Answer in natural Tanglish written in English letters when the user asks in Tanglish.
For coding questions, provide correct, runnable code that matches the exact request.
Use input() when the user asks for user input.
For even or odd, use num % 2 == 0.
Do not add unrelated conditions or features.
Explain code briefly in the user's language.
"""

examples = [
    (
        "Write a Python program to check whether a number is even or odd.",
        "num = int(input(\"Enter a number: \"))\n\nif num % 2 == 0:\n    print(\"Even\")\nelse:\n    print(\"Odd\")"
    ),
    (
        "Python-la number even-ah odd-ah check panra program kudu.",
        "num = int(input(\"Oru number enter pannu: \"))\n\nif num % 2 == 0:\n    print(\"Even\")\nelse:\n    print(\"Odd\")"
    ),
    (
        "Write a Python program to find the largest of two numbers.",
        "a = int(input(\"Enter first number: \"))\nb = int(input(\"Enter second number: \"))\n\nif a > b:\n    print(a, \"is larger\")\nelif b > a:\n    print(b, \"is larger\")\nelse:\n    print(\"Both numbers are equal\")"
    ),
    (
        "Python-la two numbers-la periya number edhu nu find panna code kudu.",
        "a = int(input(\"First number: \"))\nb = int(input(\"Second number: \"))\n\nif a > b:\n    print(a, \"perusu\")\nelif b > a:\n    print(b, \"perusu\")\nelse:\n    print(\"Rendu numbers-um equal\")"
    ),
    (
        "Write a Python program to check whether a number is positive, negative, or zero.",
        "num = float(input(\"Enter a number: \"))\n\nif num > 0:\n    print(\"Positive\")\nelif num < 0:\n    print(\"Negative\")\nelse:\n    print(\"Zero\")"
    ),
    (
        "Python-la number positive, negative illa zero-va nu check panna code kudu.",
        "num = float(input(\"Number enter pannu: \"))\n\nif num > 0:\n    print(\"Positive\")\nelif num < 0:\n    print(\"Negative\")\nelse:\n    print(\"Zero\")"
    ),
    (
        "Write a Python program to print numbers from 1 to 10.",
        "for i in range(1, 11):\n    print(i)"
    ),
    (
        "Python-la 1 irundhu 10 vara numbers print panna code kudu.",
        "for i in range(1, 11):\n    print(i)"
    ),
    (
        "Write a Python program to calculate the sum of numbers from 1 to n.",
        "n = int(input(\"Enter n: \"))\ntotal = 0\n\nfor i in range(1, n + 1):\n    total += i\n\nprint(\"Sum:\", total)"
    ),
    (
        "Python-la 1 irundhu n vara numbers-oda sum calculate panna code kudu.",
        "n = int(input(\"n value enter pannu: \"))\ntotal = 0\n\nfor i in range(1, n + 1):\n    total += i\n\nprint(\"Sum:\", total)"
    ),
    (
        "Write a Python program to calculate the factorial of a number.",
        "n = int(input(\"Enter a non-negative integer: \"))\n\nif n < 0:\n    print(\"Factorial is not defined for negative numbers\")\nelse:\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    print(\"Factorial:\", result)"
    ),
    (
        "Python-la factorial find panna program kudu.",
        "n = int(input(\"Non-negative number enter pannu: \"))\n\nif n < 0:\n    print(\"Negative number-ku factorial define pannala\")\nelse:\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    print(\"Factorial:\", result)"
    ),
    (
        "Write a Python function that returns the square of a number.",
        "def square(n):\n    return n * n\n\nnum = int(input(\"Enter a number: \"))\nprint(square(num))"
    ),
    (
        "Python-la oru number-oda square return panra function kudu.",
        "def square(n):\n    return n * n\n\nnum = int(input(\"Number enter pannu: \"))\nprint(square(num))"
    ),
    (
        "Write a Python program to count the vowels in a string.",
        "text = input(\"Enter text: \")\ncount = 0\n\nfor ch in text.lower():\n    if ch in \"aeiou\":\n        count += 1\n\nprint(\"Vowel count:\", count)"
    ),
    (
        "Python-la string-la ethana vowels irukku nu count panna code kudu.",
        "text = input(\"Text enter pannu: \")\ncount = 0\n\nfor ch in text.lower():\n    if ch in \"aeiou\":\n        count += 1\n\nprint(\"Vowels count:\", count)"
    ),
    (
        "Write a Python program to reverse a string.",
        "text = input(\"Enter text: \")\nprint(text[::-1])"
    ),
    (
        "Python-la string-ah reverse panna code kudu.",
        "text = input(\"Text enter pannu: \")\nprint(text[::-1])"
    ),
    (
        "Write a Python program to check whether a year is a leap year.",
        "year = int(input(\"Enter a year: \"))\n\nif year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):\n    print(\"Leap year\")\nelse:\n    print(\"Not a leap year\")"
    ),
    (
        "Python-la leap year check panra program kudu.",
        "year = int(input(\"Year enter pannu: \"))\n\nif year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):\n    print(\"Leap year\")\nelse:\n    print(\"Leap year illa\")"
    ),
    (
        "Write a Python program to safely convert user input to an integer.",
        "try:\n    num = int(input(\"Enter an integer: \"))\n    print(\"Number:\", num)\nexcept ValueError:\n    print(\"Please enter a valid integer.\")"
    ),
    (
        "Python-la user input integer-ah convert pannumbodhu error handle panna code kudu.",
        "try:\n    num = int(input(\"Integer enter pannu: \"))\n    print(\"Number:\", num)\nexcept ValueError:\n    print(\"Correct integer enter pannu\")"
    ),
    (
        "Write a Java program to check whether an integer is even or odd.",
        "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int num = sc.nextInt();\n\n        if (num % 2 == 0) {\n            System.out.println(\"Even\");\n        } else {\n            System.out.println(\"Odd\");\n        }\n        sc.close();\n    }\n}"
    ),
    (
        "Java-la input vaangi even or odd check panra program kudu.",
        "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int num = sc.nextInt();\n\n        if (num % 2 == 0) {\n            System.out.println(\"Even\");\n        } else {\n            System.out.println(\"Odd\");\n        }\n        sc.close();\n    }\n}"
    ),
    (
        "Write a Python program to add two numbers entered by the user.",
        "a = float(input(\"Enter first number: \"))\nb = float(input(\"Enter second number: \"))\nprint(\"Sum:\", a + b)"
    ),
    (
        "Python-la user kitta rendu numbers vaangi add panna code kudu.",
        "a = float(input(\"First number enter pannu: \"))\nb = float(input(\"Second number enter pannu: \"))\nprint(\"Sum:\", a + b)"
    ),
]

def make_record(question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    # Read the existing dataset without modifying it.
    with SOURCE.open("r", encoding="utf-8-sig") as file:
        original = [json.loads(line) for line in file if line.strip()]

    # Keep every existing example and append only new question-answer pairs.
    seen_questions = {
        record["messages"][1]["content"].strip().casefold()
        for record in original
        if "messages" in record and len(record["messages"]) >= 3
    }

    combined = list(original)
    added = 0

    for question, answer in examples:
        if question.strip().casefold() in seen_questions:
            continue

        combined.append(make_record(question, answer))
        seen_questions.add(question.strip().casefold())
        added += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        for record in combined:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Coding dataset created successfully.")
    print("Original examples:", len(original))
    print("New coding examples added:", added)
    print("Total examples:", len(combined))
    print("Output:", OUTPUT)
    print("Original dataset was not modified.")

if __name__ == "__main__":
    main()

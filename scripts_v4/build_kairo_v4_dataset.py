
import json
import re
from pathlib import Path

SOURCE = Path("data_v2/kairo_bilingual_v3.jsonl")
OUTPUT = Path("data_v4/kairo_bilingual_v4.jsonl")

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful AI assistant. "
    "Reply in the same language style as the user: clear English "
    "for English questions, natural Tanglish written in English "
    "letters for Tanglish questions. For Tanglish, explain ideas "
    "using natural spoken Tamil sentence structure with English "
    "technical terms where appropriate. Avoid keyword-only answers "
    "and unnatural word-by-word translations. Keep answers focused, "
    "accurate, and non-repetitive. Never invent facts. For code "
    "requests, follow the requested requirements exactly."
)

# Each item contains a Tanglish question and a natural Tanglish answer.
EXAMPLES = [
    (
        "Machine learning na enna? Simple-ah sollu.",
        "Machine learning na, computer-ku data-la irukkura patterns-ai "
        "learn panna solli train panradhu. Andha patterns use panni "
        "pudhu data pathi prediction panna mudiyum. Example: spam "
        "message-ah identify panradhu."
    ),
    (
        "Deep learning na enna?",
        "Deep learning na machine learning-oda oru type. Idhula neural "
        "network-nu sollra model use panni images, audio, text madhiri "
        "complex data-la irundhu patterns learn pannuvanga. Example: "
        "photo-la irukkura objects-ai identify panradhu."
    ),
    (
        "AI kum machine learning kum enna difference?",
        "AI na computer intelligent-ah tasks perform panra periya field. "
        "Machine learning adhulaye oru part; data-la irundhu patterns "
        "learn panni prediction panna use aagum. Simple-ah sonna, "
        "machine learning enbadhu AI-ai create panna use panra oru method."
    ),
    (
        "Cloud computing na enna? Example kudu.",
        "Cloud computing na namma computer-la ellathayum store pannama, "
        "internet moolama vera servers-la irukkura storage and computing "
        "resources-ai use panradhu. Example: Google Drive-la files save "
        "panni enga irundhaalum access pannalaam."
    ),
    (
        "Database na enna?",
        "Database na information-ai organized-ah store panni manage panra "
        "place. Thevai padumbodhu data-va easy-ah search pannalaam, "
        "update pannalaam. Example: college student names and marks "
        "store panra system."
    ),
    (
        "API na enna simple-ah explain pannu.",
        "API na rendu software applications onnu innonnu kooda communicate "
        "panna use panra interface. Example: weather app, weather service "
        "API-kitta temperature details ketu display pannalaam."
    ),
    (
        "Python-la variable na enna?",
        "Variable na oru value-ai store panna use panra name. Example, "
        "age = 20 nu ezhudhina, age-nu oru variable-la 20 store aagum. "
        "Adha program-la thevai padumbodhu use pannalaam."
    ),
    (
        "Loop na enna Python-la?",
        "Loop na ore set of instructions-ai repeated-ah execute panna "
        "use panradhu. Example, 1-lendhu 5 vara numbers print pannanumna, "
        "ovvoru number-kum separate line ezhudhaama for loop use pannalaam."
    ),
    (
        "Function na enna programming-la?",
        "Function na oru specific task panna ezhudhra reusable code block. "
        "Adhai thevai padumbodhu call pannalaam. Idhanaala same code-ai "
        "thirumba thirumba ezhudha vendiya avasiyam kuraiyum."
    ),
    (
        "Python-la list na enna?",
        "List na multiple values-ai ore variable-la order-ah store panna "
        "use panra data structure. Example: fruits = ['apple', 'mango']. "
        "List-la items-ai access pannalaam, add pannalaam, remove pannalaam."
    ),
    (
        "Error vandha first enna check pannanum?",
        "First error message-la enna sollirukku-nu paaru. Appuram line "
        "number, variable names, brackets, indentation ellam check pannu. "
        "Error message-ai full-ah paatha problem enga irukku-nu kandupidikka "
        "easy-ah irukkum."
    ),
    (
        "Overfitting na enna machine learning-la?",
        "Overfitting na model training data-vai romba closely learn pannidum. "
        "Aana pudhu data kudukkumbodhu correct-ah perform pannaama pogalaam. "
        "Idhai reduce panna more varied data, regularization, and validation "
        "use pannalaam."
    ),
    (
        "Data structure ethukku use aagum?",
        "Data structure na data-ai efficient-ah store panni access panna "
        "help panra method. Array, list, stack, queue, tree ellam examples. "
        "Correct data structure select panna program speed and memory usage "
        "improve aagum."
    ),
    (
        "Git ethukku use panrom?",
        "Git na code changes-ai track panna use panra version control system. "
        "Yaar enna change pannanga-nu paakalaam, previous version-ku "
        "thirumba pogalaam, team-ah work pannumbodhu changes-ai manage "
        "pannalaam."
    ),
    (
        "Frontend kum backend kum difference enna?",
        "Frontend na user screen-la paakura buttons, forms, layout madhiri "
        "parts. Backend na data processing, business logic, database "
        "communication madhiri server-side work. Rendume serndhudhaan "
        "complete web application work aagum."
    ),
    (
        "SQL na enna?",
        "SQL na relational database-la data-va query panna and manage panna "
        "use panra language. Idha use panni records retrieve, insert, "
        "update, delete pannalaam. Example: students table-la irukkura "
        "ella records-ai SELECT query moolama edukkalaam."
    ),
    (
        "Cybersecurity na enna?",
        "Cybersecurity na computers, networks, applications, and data-va "
        "unauthorized access and attacks-lendhu protect panradhu. Strong "
        "password, software updates, and multi-factor authentication "
        "madhiri practices security-ai improve panna help pannum."
    ),
    (
        "Neural network na enna?",
        "Neural network na data-la irukkura patterns-ai learn panna use "
        "panra machine learning model. Idhula interconnected layers "
        "irukkum. Training time-la model predictions-oda errors-ai reduce "
        "panna weights-ai adjust pannum."
    ),
    (
        "RAM kum storage kum enna difference?",
        "RAM na computer run aagumbodhu currently use panra data-va "
        "temporary-ah store pannum; power off aana adhu clear aagum. "
        "SSD madhiri storage-la files long-term-ah save aagum, power off "
        "aanaalum data irukkum."
    ),
    (
        "API use panra oru example sollu.",
        "Suppose oru food delivery app-la payment panna, andha app payment "
        "service API-ai call pannalaam. API request-ai process panni "
        "payment success-a fail-a nu response tharum. App adha user-kku "
        "display pannum."
    ),
    (
        "Python-la even odd check panra program kudu.",
        "Indha program number-ai input-ah vaangi, 2-la divide pannumbodhu "
        "remainder zero-va irukkaa-nu check pannum.\n\n"
        "```python\n"
        "num = int(input('Enter a number: '))\n"
        "if num % 2 == 0:\n"
        "    print('Even')\n"
        "else:\n"
        "    print('Odd')\n"
        "```"
    ),
    (
        "Factorial na enna?",
        "Oru positive integer-oda factorial na, andha number-lendhu "
        "1 vara ella integers-ai multiply panradhu. Example: 5! = "
        "5 × 4 × 3 × 2 × 1 = 120. Special-ah 0! = 1."
    ),
    (
        "Recursion na enna?",
        "Recursion na oru function thannaiye call panni problem-ai smaller "
        "parts-ah solve panra technique. Aana eppodhu stop pannanum-nu "
        "base case kandippa irukkanum; illaina function endless-ah call "
        "aagalaam."
    ),
    (
        "Operating system enna seiyum?",
        "Operating system na hardware-um applications-um work panna "
        "manage panra software. Idhu memory, files, processes, and devices-ai "
        "manage pannum. Windows, Linux, Android ellam operating system "
        "examples."
    ),
    (
        "Model training loss kammi aana model nalla irukkuma?",
        "Training loss kammi aaguradhu model training examples-ai better-ah "
        "fit pannudhu-nu indicate pannalaam. Aana adhu mattum podhaadhu. "
        "Unseen validation data, factual accuracy, and actual task tests "
        "moolama quality-ai evaluate pannanum."
    ),
    (
        "Local AI model na enna?",
        "Local AI model na un computer-la run aagura AI model. Internet "
        "illaama supported tasks-ai perform panna mudiyum. Aana speed and "
        "answer quality un hardware, model size, and training-ai poruthadhu."
    ),
    (
        "AI hallucination na enna?",
        "AI hallucination na model confident-ah oru answer sollum, aana "
        "andha answer thappa irukkalaam illa fact-ku support illaama "
        "irukkalaam. Adhanaala important information-ai reliable sources "
        "moolama verify pannanum."
    ),
    (
        "Python code correct-ah irukkaa nu eppadi test panradhu?",
        "First syntax check pannu. Appuram normal input mattum illaama "
        "zero, negative values, empty input madhiri relevant edge cases-ai "
        "test pannu. Expected output-oda actual output compare pannina "
        "logic errors kandupidikka mudiyum."
    ),
    (
        "Internet illama local AI use panna mudiyuma?",
        "Aama, model files already download aagi irundha local AI-ai "
        "internet illama run panna mudiyum. Aana online information "
        "thevai padra questions-ku local model-kitta latest details "
        "irukkaadhu."
    ),
    (
        "Dataset-la duplicate examples irundha enna problem?",
        "Duplicate examples adhigama irundha model sila patterns-ai "
        "over-learn pannalaam; dataset diversity-um kuraiyum. Adhanaala "
        "duplicate questions-ai check panni, different topics and "
        "wording-la quality examples add panradhu useful."
    ),
]


def normalize(text):
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    seen_questions = set()
    duplicate_count = 0

    # Preserve all existing valid records.
    with SOURCE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                )

            messages = record.get("messages", [])
            user_messages = [
                message.get("content", "")
                for message in messages
                if message.get("role") == "user"
            ]

            if not user_messages:
                print(
                    f"Skipping record without user message: "
                    f"line {line_number}"
                )
                continue

            key = normalize(user_messages[-1])

            if key in seen_questions:
                duplicate_count += 1
                continue

            seen_questions.add(key)
            records.append(record)

    original_count = len(records)
    added_count = 0

    for question, answer in EXAMPLES:
        key = normalize(question)

        if key in seen_questions:
            duplicate_count += 1
            continue

        record = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]
        }

        records.append(record)
        seen_questions.add(key)
        added_count += 1

    with OUTPUT.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Verify the saved JSONL file.
    verified_count = 0

    with OUTPUT.open("r", encoding="utf-8") as file:
        for line in file:
            json.loads(line)
            verified_count += 1

    print("\nKAIRO v4 dataset created successfully.")
    print(f"Original examples preserved: {original_count}")
    print(f"New Tanglish examples added: {added_count}")
    print(f"Duplicate examples skipped: {duplicate_count}")
    print(f"Total examples: {verified_count}")
    print(f"Output file: {OUTPUT}")
    print("\nExisting v3 dataset was not modified.")
    print("Dataset creation completed. Training has NOT started.")


if __name__ == "__main__":
    main()

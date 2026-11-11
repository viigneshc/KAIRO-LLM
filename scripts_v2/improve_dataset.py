
import json
from pathlib import Path

SOURCE = Path("data_v2/kairo_bilingual_v1.jsonl")
BACKUP = Path("data_v2/kairo_bilingual_v1_backup.jsonl")
OUTPUT = Path("data_v2/kairo_bilingual_v2.jsonl")

SYSTEM = """You are KAIRO, a helpful AI assistant.
Answer in the same language style as the user.
For English questions, use clear, simple English.
For Tanglish questions, use natural spoken Tamil written in English letters, mixed with common English words.
Do not use Tamil script when answering Tanglish.
Keep answers accurate, useful, and easy to understand.
Avoid awkward invented words, unnecessary repetition, and long introductions.
For coding questions, provide correct code and follow the user's requirements."""

# Each pair covers the same concept in English and natural Tanglish.
pairs = [
    (
        "What is cloud computing?",
        "Cloud computing na enna?",
        "Cloud computing means using servers, storage, databases, and other computing resources over the internet.",
        "Cloud computing na internet moolama servers, storage, databases madhiri computing resources-ah use panradhu. Own server maintain panna thevai kuraiyum."
    ),
    (
        "What is machine learning?",
        "Machine learning na enna?",
        "Machine learning is a part of AI where computers learn patterns from data to make predictions or decisions.",
        "Machine learning na AI-oda oru part. Idhula computer data-la irundhu patterns-ah learn panni predictions illa decisions edukka kathukkum."
    ),
    (
        "What is artificial intelligence?",
        "Artificial intelligence na enna?",
        "Artificial intelligence is a field of computing that enables machines to perform tasks that usually require human intelligence.",
        "Artificial intelligence na, humans intelligence use panni panra tasks-ah machines perform panna help panra technology."
    ),
    (
        "What is deep learning?",
        "Deep learning na enna?",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns.",
        "Deep learning na machine learning-oda oru type. Idhula multiple layers irukkura neural networks use panni complex patterns-ah learn pannuvanga."
    ),
    (
        "What is a database?",
        "Database na enna?",
        "A database is an organized collection of data that can be stored, searched, updated, and managed efficiently.",
        "Database na data-va organized-ah store panni, easy-ah search, update, manage panna use panra system."
    ),
    (
        "What is an API?",
        "API na enna?",
        "An API allows two software applications to communicate and exchange data through defined rules.",
        "API na rendu software applications onnu kooda onnu communicate panni data exchange panna help panra interface."
    ),
    (
        "What is cybersecurity?",
        "Cybersecurity na enna?",
        "Cybersecurity protects computers, networks, and data from unauthorized access, attacks, and damage.",
        "Cybersecurity na computers, networks, data ellathayum unauthorized access-um cyber attacks-um irundhu protect panradhu."
    ),
    (
        "What is a programming language?",
        "Programming language na enna?",
        "A programming language is used to write instructions that a computer can execute.",
        "Programming language na computer enna task pannanum-nu instructions write panna use panra language."
    ),
    (
        "What is an operating system?",
        "Operating system na enna?",
        "An operating system manages a computer's hardware and software and provides services for applications.",
        "Operating system na computer hardware-um software-um manage panni, applications run aaga help panra system."
    ),
    (
        "What is a computer network?",
        "Computer network na enna?",
        "A computer network connects devices so they can communicate and share data or resources.",
        "Computer network na devices onnu kooda onnu connect aagi data-um resources-um share pannikka help panra setup."
    ),
    (
        "What is an algorithm?",
        "Algorithm na enna?",
        "An algorithm is a step-by-step procedure for solving a problem or completing a task.",
        "Algorithm na oru problem-ah solve panna illa task complete panna follow panra step-by-step procedure."
    ),
    (
        "What is an HTML tag?",
        "HTML tag na enna?",
        "An HTML tag defines an element on a webpage, such as a heading, paragraph, image, or link.",
        "HTML tag na webpage-la heading, paragraph, image, link madhiri elements-ah define panna use panradhu."
    ),
    (
        "What is a variable in programming?",
        "Programming-la variable na enna?",
        "A variable is a named storage location used to hold a value that a program can access or change.",
        "Variable na program-la oru value-va store panna use panra named storage location. Thevai pattina andha value-va change pannalam."
    ),
    (
        "What is a loop in programming?",
        "Programming-la loop na enna?",
        "A loop repeats a block of code while a specified condition is satisfied or for a chosen number of times.",
        "Loop na oru code block-ah condition satisfy aagura varaikkum illa specified number of times repeat panna use panradhu."
    ),
    (
        "What is an IP address?",
        "IP address na enna?",
        "An IP address is a numerical address used to identify a device on an IP network.",
        "IP address na network-la oru device-ah identify panna use panra numerical address."
    ),
    (
        "What is a website frontend?",
        "Frontend na enna?",
        "The frontend is the part of a website that users see and interact with, usually built using HTML, CSS, and JavaScript.",
        "Frontend na website-la users paakura and interact panra part. Idha usually HTML, CSS, JavaScript use panni create pannuvanga."
    ),
    (
        "What is a backend?",
        "Backend na enna?",
        "The backend handles application logic, data processing, authentication, and communication with databases or other services.",
        "Backend na application-oda internal logic, data processing, login authentication, database communication madhiri work-ah handle panra part."
    ),
    (
        "What is version control?",
        "Version control na enna?",
        "Version control tracks changes to files over time and helps developers collaborate and restore earlier versions.",
        "Version control na files-la nadakkura changes-ah track pannum. Developers collaborate pannavum old versions-ku thirumba pogavum help pannum."
    ),
    (
        "What is an SQL query?",
        "SQL query na enna?",
        "An SQL query is an instruction used to retrieve, insert, update, or delete data in a database.",
        "SQL query na database-la irundhu data retrieve panna, insert panna, update panna, delete panna use panra instruction."
    ),
    (
        "What is an exception in Python?",
        "Python-la exception na enna?",
        "An exception is an error or unexpected event that occurs while a program is running and can often be handled with try and except.",
        "Exception na program run aagumbodhu varra error illa unexpected event. Python-la try-except use panni adha handle pannalam."
    ),
    (
        "How can I improve my programming skills?",
        "Programming skills-ah epdi improve panradhu?",
        "Practice coding regularly, solve problems, build small projects, read error messages, and review your code.",
        "Programming skills improve panna daily coding practice pannu, problems solve pannu, small projects build pannu, errors-ah understand panni code review pannu."
    ),
    (
        "Explain cloud computing simply.",
        "Cloud computing-ah simple-ah explain pannu.",
        "Cloud computing lets you use computing services over the internet instead of managing all the hardware yourself.",
        "Simple-ah sonna, un computer-la ellathayum store pannaama internet moolama storage, servers madhiri services use panradhu dhaan cloud computing."
    ),
    (
        "What is data science?",
        "Data science na enna?",
        "Data science uses data analysis, statistics, and programming to find useful patterns and support decisions.",
        "Data science na data-va analyze panni useful patterns kandupidichu decisions edukka help panra field. Idhula statistics-um programming-um use pannuvanga."
    ),
    (
        "What is a neural network?",
        "Neural network na enna?",
        "A neural network is a machine learning model made of connected units that learn patterns from examples.",
        "Neural network na connected units use panni examples-la irundhu patterns learn panra machine learning model."
    ),
    (
        "What is a software bug?",
        "Software bug na enna?",
        "A software bug is a flaw that causes a program to behave incorrectly or produce an unexpected result.",
        "Software bug na program-la irukkura mistake. Adhanaala program thappa work pannalam illa unexpected result varalam."
    ),
]

def make_example(question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE}")

    original = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    # Preserve the source and create a backup only if one does not exist.
    if not BACKUP.exists():
        BACKUP.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in original) + "\n",
            encoding="utf-8"
        )

    # Build a separate v2 dataset: old examples plus bilingual pairs.
    combined = list(original)
    seen = {
        (
            r["messages"][1]["content"].strip().casefold(),
            r["messages"][2]["content"].strip().casefold()
        )
        for r in combined
        if len(r.get("messages", [])) == 3
    }

    added = 0
    for english_q, tanglish_q, english_a, tanglish_a in pairs:
        for q, a in [
            (english_q, english_a),
            (tanglish_q, tanglish_a),
        ]:
            key = (q.casefold(), a.casefold())
            if key not in seen:
                combined.append(make_example(q, a))
                seen.add(key)
                added += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in combined) + "\n",
        encoding="utf-8"
    )

    print("Original examples:", len(original))
    print("New examples added:", added)
    print("Expanded dataset:", len(combined))
    print("Saved to:", OUTPUT)
    print("Original dataset unchanged:", SOURCE.exists())

if __name__ == "__main__":
    main()

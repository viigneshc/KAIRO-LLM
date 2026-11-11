import json
from pathlib import Path

data = [
    ("What is artificial intelligence?", "Artificial intelligence (AI) enables computers to perform tasks that normally require human intelligence, such as learning, reasoning, and understanding language.",
     "Artificial intelligence na enna?", "Artificial Intelligence (AI) na, human intelligence thevai padura tasks-ah computer perform panna help panra technology. Example: language purinjukaradhu, learn panradhu, problems solve panradhu."),
    ("What is machine learning?", "Machine learning is a branch of AI in which computers learn patterns from data and use them to make predictions or decisions.",
     "Machine learning na enna?", "Machine learning na AI-oda oru part. Idhula computer data-la irundhu patterns-ah learn panni, predictions illa decisions edukka kathukkum."),
    ("What is deep learning?", "Deep learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns.",
     "Deep learning na enna?", "Deep learning na machine learning-oda oru type. Idhu multiple layers irukkura neural networks use panni complex patterns-ah learn pannum."),
    ("What is cloud computing?", "Cloud computing provides computing resources such as servers, storage, and databases over the internet.",
     "Cloud computing na enna?", "Cloud computing na internet moolama servers, storage, databases madhiri computing resources-ah use panradhu. Own server maintain panna thevai kuraiyum."),
    ("What is a database?", "A database is an organised collection of data that can be stored, accessed, updated, and managed efficiently.",
     "Database na enna?", "Database na data-va organised-ah store panni, thevai padumbodhu retrieve, update, manage panna use aagura system."),
    ("What is Python?", "Python is a high-level programming language known for its readable syntax. It is used in automation, web development, data science, and AI.",
     "Python na enna?", "Python na easy-ah read panni learn panna mudiyura programming language. Idha automation, web development, data science, AI madhiri areas-la use pannuvanga."),
    ("What is Java?", "Java is a programming language commonly used to build applications. It runs on compatible Java Virtual Machine environments.",
     "Java na enna?", "Java na applications develop panna use panra programming language. Java Virtual Machine support irukkura environment-la idhu run aagum."),
    ("What is an algorithm?", "An algorithm is a finite sequence of clear steps used to solve a problem or complete a task.",
     "Algorithm na enna?", "Algorithm na oru problem-ah solve panna follow panra clear step-by-step procedure."),
    ("How do I debug a program?", "Read the error message, locate the relevant line, reproduce the problem, and test one change at a time.",
     "Program-la error vandha epdi debug panradhu?", "First error message-ah read pannu. Error endha line-la varudhunu identify pannu. Problem-ah reproduce panni, oru change-ah mattum test pannu."),
    ("How can I learn programming?", "Start with variables, conditions, loops, and functions. Practise small problems daily, then build projects and learn debugging.",
     "Programming epdi learn panradhu?", "First variables, conditions, loops, functions kathukko. Daily small problems solve pannu. Apram projects build panni debugging practice pannu."),
    ("What is a neural network?", "A neural network is a machine learning model made of connected units called neurons. It learns patterns by adjusting weights during training.",
     "Neural network na enna?", "Neural network na connected artificial neurons use panra machine learning model. Training time-la weights-ah adjust panni data patterns-ah learn pannum."),
    ("What is cybersecurity?", "Cybersecurity protects computers, networks, applications, and data from unauthorised access, attacks, and damage.",
     "Cybersecurity na enna?", "Cybersecurity na computers, networks, applications, data ellathayum unauthorised access, attacks, damage-la irundhu protect panradhu."),
    ("What is an API?", "An API defines how different software applications communicate and exchange data with each other.",
     "API na enna?", "API na rendu software applications epdi communicate panni data exchange pannanum nu define panra interface."),
    ("What is HTML?", "HTML is the markup language used to structure content on web pages, including headings, paragraphs, links, and images.",
     "HTML na enna?", "HTML na web page-oda structure create panna use panra markup language. Headings, paragraphs, links, images ellam add pannalaam."),
    ("What is CSS?", "CSS controls the appearance and layout of web pages, including colours, spacing, fonts, and responsive design.",
     "CSS na enna?", "CSS na web page eppadi theriyanum nu control pannum. Colours, spacing, fonts, layout ellam style panna use pannuvom."),
    ("What is JavaScript?", "JavaScript adds behaviour and interactivity to web pages, such as button actions, form validation, and dynamic updates.",
     "JavaScript na enna?", "JavaScript na web page-ku interactivity add panna use aagum. Example: button click actions, form validation, dynamic updates."),
    ("What is SQL?", "SQL is a language used to query and manage data in relational database systems.",
     "SQL na enna?", "SQL na relational database-la irukkura data-va retrieve, insert, update, delete panna use panra language."),
    ("What is overfitting in machine learning?", "Overfitting happens when a model learns training data too closely, including noise, and performs poorly on unseen data.",
     "Machine learning-la overfitting na enna?", "Overfitting na model training data-va romba closely learn pannidum. Aana pudhu data kudukkumbodhu correct-ah perform pannaama pogalaam."),
    ("What is the difference between AI and machine learning?", "AI is the broader field of creating systems that perform intelligent tasks. Machine learning is a subset of AI that learns patterns from data.",
     "AI-க்கும் machine learning-க்கும் enna difference?", "AI na intelligent tasks perform panra systems create panra broad field. Machine learning na data-la irundhu learn panra AI-oda oru part."),
    ("What is an operating system?", "An operating system manages computer hardware and software resources and provides services for applications.",
     "Operating system na enna?", "Operating system na computer hardware, software resources-ah manage panni applications run aaga thevaiyana services provide pannum."),
    ("What is an IP address?", "An IP address is a numerical address used to identify a device on an IP network.",
     "IP address na enna?", "IP address na network-la oru device-ah identify panna use panra numerical address."),
    ("What is version control?", "Version control tracks changes to files over time and helps developers collaborate and restore earlier versions.",
     "Version control na enna?", "Version control na files-la nadakkura changes-ah track pannum. Developers collaborate pannavum, old versions-ku thirumba pogavum help pannum."),
    ("Explain photosynthesis simply.", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce sugar and release oxygen.",
     "Photosynthesis-ah simple-ah explain pannu.", "Photosynthesis na plants sunlight, water, carbon dioxide use panni food (sugar) prepare panra process. Indha process-la oxygen release aagum."),
    ("What is data science?", "Data science combines statistics, programming, and domain knowledge to analyse data and extract useful insights.",
     "Data science na enna?", "Data science na statistics, programming, domain knowledge use panni data-ah analyse seithu useful insights kandupidikkura field."),
    ("What is a REST API?", "A REST API lets applications communicate over HTTP using resources and methods such as GET, POST, PUT, and DELETE.",
     "REST API na enna?", "REST API na HTTP moolama applications communicate panna use aagum. GET, POST, PUT, DELETE madhiri methods use panni data access illa update pannalaam."),
    ("How do I write a good function?", "Give the function one clear responsibility, choose a descriptive name, use meaningful parameters, and handle expected edge cases.",
     "Nalla function epdi write panradhu?", "Oru function-ku clear-ah oru responsibility kudu. Meaningful name, correct parameters use pannu. Possible edge cases-um handle pannu."),
    ("Why should I learn data structures?", "Data structures organise information for efficient access and modification. Choosing the right structure can improve program performance.",
     "Data structures yen learn pannanum?", "Data structures data-va efficient-ah organise panna help pannum. Correct data structure choose panna program performance improve aagum."),
    ("What is a computer network?", "A computer network connects devices so they can communicate and share information or resources.",
     "Computer network na enna?", "Computer network na devices connect aagi information-um resources-um share pannikka help panra system."),
    ("What is an SQL injection attack?", "SQL injection is an attack in which untrusted input changes a database query. Parameterised queries help prevent it.",
     "SQL injection na enna?", "SQL injection na user input moolama database query-ah attacker manipulate panra attack. Parameterised queries use pannina indha risk-ah reduce pannalaam."),
    ("What is Git?", "Git is a distributed version control system that tracks code changes and supports collaboration between developers.",
     "Git na enna?", "Git na code changes-ah track panna use panra distributed version control system. Team members collaborate pannavum help pannum."),
    ("What is responsive web design?", "Responsive web design adapts a website layout to different screen sizes, including phones, tablets, and desktops.",
     "Responsive web design na enna?", "Responsive web design na website mobile, tablet, desktop madhiri different screen sizes-ku adapt aagura maadhiri design panradhu."),
    ("How should I approach a coding problem?", "Understand the input and output, identify constraints, create a simple approach, test edge cases, and then optimise if needed.",
     "Coding problem-ah epdi approach panradhu?", "First input, output, constraints purinjikko. Simple solution yosichu, edge cases test pannu. Thevai irundha apram optimise pannu."),
    ("What is machine learning training data?", "Training data is the collection of examples a machine learning model uses to learn patterns and adjust its parameters.",
     "Training data na enna?", "Training data na machine learning model patterns learn panni parameters adjust panna use panra examples collection."),
    ("What is the difference between frontend and backend?", "Frontend is the part users interact with. Backend handles application logic, data processing, and server-side operations.",
     "Frontend-க்கும் backend-க்கும் enna difference?", "Frontend na user direct-ah paathu interact panra part. Backend na application logic, data processing, server-side operations handle panra part."),
    ("How can I make code easier to maintain?", "Use clear names, small functions, consistent formatting, useful comments, and tests. Avoid unnecessary duplication.",
     "Code maintain panna easy-ah epdi write panradhu?", "Clear names, small functions, consistent formatting use pannu. Useful comments and tests add pannu. Same code-ah thevai illaama repeat pannaadha."),
    ("What should an AI assistant do when unsure?", "It should state uncertainty clearly, avoid inventing facts, and ask a focused question or explain how to verify the answer.",
     "AI assistant-ku answer theriyalana enna pannanum?", "Answer sure-ah theriyalana adha clear-ah sollanum. Facts invent panna koodadhu. Thevai na oru focused question kekkalaam illa verify panra method sollalaam."),
    ("Explain binary search.", "Binary search finds a target in sorted data by repeatedly checking the middle element and discarding half of the remaining search range.",
     "Binary search na enna?", "Binary search na sorted data-la target value-ah theduvadhu. Middle element check panni, thevai illaadha half-ah remove pannitu search continue pannuvom."),
    ("What is a software bug?", "A software bug is a defect that causes a program to behave incorrectly or unexpectedly.",
     "Software bug na enna?", "Software bug na program expected-ah work aagaama, wrong illa unexpected result kudukka kaaranama irukkura defect."),
    ("What is an online public AI service?", "An online AI service runs on a remote server and lets users interact with it through an application or website over the internet.",
     "Online AI service na enna?", "Online AI service na remote server-la run aagum. Users internet moolama website illa application use panni adha access pannalaam."),
    ("What is an offline AI model?", "An offline AI model runs on a local device after its required files are available. It can work without internet, within the limits of the device.",
     "Offline AI model na enna?", "Offline AI model na required model files device-la irundha, internet illaamal local-ah run panna mudiyum. Aana speed and quality device capacity-ah depend pannum."),
    ("What is a good way to explain a difficult topic?", "Start with the main idea, use simple language, give a relatable example, and then explain important details.",
     "Kashtamaana topic-ah easy-ah epdi explain panradhu?", "First main idea-ah simple-ah sollu. Daily life example kudu. Apram important details-ah step-by-step explain pannu."),
    ("What is a programming loop?", "A loop repeats a block of code while a condition is satisfied or for a specified number of iterations.",
     "Programming-la loop na enna?", "Loop na oru condition satisfy aagura varaikkum, illa specified number of times, code block-ah repeat panna use pannuvom."),
    ("What is an exception in programming?", "An exception is an event that interrupts normal program execution. Programs can handle expected exceptions to avoid crashing.",
     "Programming-la exception na enna?", "Exception na program normal-ah run aaguradha interrupt panra event. Expected exceptions-ah handle pannina program crash aaguradha avoid pannalaam."),
    ("How do I improve my English?", "Practise reading, listening, speaking, and writing regularly. Learn useful phrases and review mistakes without worrying about perfection.",
     "English epdi improve panradhu?", "Daily reading, listening, speaking, writing practice pannu. Useful phrases kathukko. Mistakes-ah review pannu; starting-la perfect-ah pesa try panni bayapada thevai illa."),
    ("How can I prepare for a technical interview?", "Review fundamentals, practise coding problems, explain your projects clearly, and rehearse answering questions aloud.",
     "Technical interview-ku epdi prepare panradhu?", "Fundamentals revise pannu, coding problems practice pannu, un projects-ah clear-ah explain panna prepare aagu. Questions-ku aloud-ah answer solli practice pannu."),
    ("What is KAIRO?", "KAIRO is a personal AI assistant project designed to help with learning, programming, and everyday questions.",
     "KAIRO na enna?", "KAIRO na learning, programming, everyday questions-la help panna design panra personal AI assistant project."),
]

rows = []
for en_q, en_a, ta_q, ta_a in data:
    for question, answer in ((en_q, en_a), (ta_q, ta_a)):
        rows.append({
            "messages": [
                {"role": "system", "content": "You are KAIRO, a helpful AI assistant. Reply in the same language style as the user: clear English for English questions, natural Tanglish written in English letters for Tanglish questions. Keep answers focused, accurate, and non-repetitive. Never invent facts. For code requests, follow the requested requirements exactly."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })

path = Path("data_v2/kairo_bilingual_v1.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Created:", path)
print("Training examples:", len(rows))
print("Language pairs:", len(data))

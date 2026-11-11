
import ast
import operator
import re
import textwrap
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================================================
# CONFIGURATION
# =====================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "output/kairo-lora"

MAX_NEW_TOKENS = 450
MAX_INPUT_TOKENS = 3072
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """
You are KAIRO, a general-purpose AI assistant.

ACCURACY:
- Answer the exact question asked.
- Never invent facts, calculations, or sources.
- If unsure, say so clearly.
- Do not claim internet access or code execution unless it happened.
- Follow requested formats as closely as possible.

LANGUAGE RULES:
- Identify the language requested by the user.
- If the user requests Tanglish, answer ONLY in natural Tanglish.
- Tanglish means Tamil language written using English letters.
- Do not use Hindi, Urdu, or unrelated languages.
- Do not translate the answer into formal English.
- Keep technical terms such as cloud, server, database, Python,
  machine learning, and internet in English where appropriate.
- Example:
  "Cloud computing na internet moolama servers, storage,
  database maadhiri resources use panradhu.
  Namma own server vaangaama cloud services use pannalaam."
- Never use Hindi phrases such as "kya hai", "karna", "ko bhi",
  "hai", or "mein" in a Tanglish answer.
- Use simple, natural Tamil written in English letters.

PROGRAMMING:
- Answer harmless educational coding questions.
- Provide complete runnable code when requested.
- Include required imports and input handling.
- Explain important edge cases.
- Do not claim code was executed unless it was.

GENERAL ANSWERS:
- Start with a direct answer.
- Use examples when helpful.
- Avoid unnecessary repetition.
- Ask for clarification when necessary.
- For current information, explain that offline knowledge may be outdated.
"""


# =====================================================
# SAFE ARITHMETIC
# =====================================================

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression):
    if len(expression) > 150:
        raise ValueError("Expression too long")

    tree = ast.parse(expression, mode="eval")

    def visit(node, depth=0):
        if depth > 30:
            raise ValueError("Expression too complex")

        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Invalid number")
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in BINARY_OPERATORS:
                raise ValueError("Unsupported operation")

            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)

            result = BINARY_OPERATORS[op_type](left, right)

            if isinstance(result, (int, float)):
                if abs(result) > 1e100:
                    raise ValueError("Result too large")

            return result

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in UNARY_OPERATORS:
                raise ValueError("Unsupported operation")

            return UNARY_OPERATORS[op_type](
                visit(node.operand, depth + 1)
            )

        raise ValueError("Unsupported expression")

    return visit(tree)


def find_arithmetic_expression(question):
    text = question.lower().replace(",", "")
    text = text.replace("×", "*").replace("÷", "/")

    replacements = [
        (r"\b(multiplied\s+by|times)\b", "*"),
        (r"\b(divided\s+by)\b", "/"),
        (r"\b(plus|added\s+to)\b", "+"),
        (r"\b(minus|subtract)\b", "-"),
        (r"\b(modulo|mod)\b", "%"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*([+*/%\-])\s*(\d+(?:\.\d+)?)",
        text
    )

    if match:
        left, symbol, right = match.groups()
        return f"{left} {symbol} {right}"

    candidate = text.strip().rstrip("?=.")

    if re.fullmatch(r"[\d\s.+*/()%\-]+", candidate):
        if re.search(r"[+*/%\-]", candidate):
            return candidate

    return None


def verify_math(question):
    expression = find_arithmetic_expression(question)

    if expression is None:
        return None

    try:
        result = safe_calculate(expression)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return (
            "Verified calculation (Python):\n"
            f"{expression} = {result}\n"
            "Arithmetic result calculated directly."
        )

    except (
        ValueError, SyntaxError, ZeroDivisionError,
        OverflowError, TypeError
    ):
        return None


# =====================================================
# PROGRAMMING TEMPLATE
# =====================================================

PRIME_PROGRAM = '''```python
def is_prime(n):
    if n <= 1:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    divisor = 3

    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2

    return True


number = int(input("Enter a number: "))

if is_prime(number):
    print(number, "is a prime number")
else:
    print(number, "is not a prime number")
```'''


def handle_known_programming_request(question):
    text = question.lower()

    prime_terms = [
        "prime number",
        "check prime",
        "prime program",
        "is prime",
    ]

    asks_for_code = any(
        word in text
        for word in [
            "program", "code", "python", "write",
            "create", "check", "kudu", "give"
        ]
    )

    if asks_for_code and any(term in text for term in prime_terms):
        if any(word in text for word in ["tanglish", "tunglish"]):
            explanation = (
                "Indha Python program oru number prime-ah illaya-nu "
                "check pannum. 1-kku keezha irukkura numbers prime illa. "
                "Divisors-ai square root varaikkum check pannum."
            )
        else:
            explanation = (
                "This Python program checks whether a number is prime. "
                "Numbers less than or equal to 1 are not prime. "
                "It checks divisors up to the square root of the number."
            )

        return explanation + "\n\n" + PRIME_PROGRAM

    return None


# =====================================================
# TANGlish DETECTION AND COMMON VERIFIED EXPLANATIONS
# =====================================================

def asks_for_tanglish(question):
    text = question.lower()

    indicators = [
        "tanglish", "tunglish", "tunglish-la",
        "tanglish-la", "explain pannu", "enna?",
        "na enna", "epdi", "eppadi", "kudu",
        "solu", "sollu", "puriyura maari",
    ]

    return any(item in text for item in indicators)


def handle_known_tanglish_question(question):
    text = question.lower()

    cloud_terms = [
        "cloud computing",
        "cloud computing na enna",
    ]

    if any(term in text for term in cloud_terms):
        return (
            "Cloud computing na internet moolama servers, storage, "
            "database maadhiri computing resources use panradhu.\n\n"
            "Example: Namma laptop-la files mattum store pannaama, "
            "Google Drive maadhiri cloud service-la store pannalaam. "
            "Internet irundha vera device-layum access pannalaam.\n\n"
            "Simple-ah sonna, own computer-la ellathayum maintain "
            "pannaama, internet moolama vera servers-oda resources "
            "use panradhu dhaan cloud computing."
        )

    return None


# =====================================================
# PYTHON SYNTAX VALIDATION
# Syntax check only; never executes generated code.
# =====================================================

def extract_python_blocks(answer):
    return re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        answer,
        flags=re.IGNORECASE | re.DOTALL
    )


def validate_python_code(answer):
    blocks = extract_python_blocks(answer)

    if not blocks:
        return None

    results = []

    for index, block in enumerate(blocks, start=1):
        code = textwrap.dedent(block).strip()

        try:
            ast.parse(code)
            results.append(
                f"Python block {index}: syntax check passed."
            )
        except SyntaxError as error:
            results.append(
                f"Python block {index}: syntax error on line "
                f"{error.lineno}: {error.msg}."
            )

    return "\n".join(results)


# =====================================================
# MODEL LOADING
# =====================================================

def load_kairo():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    print("\n" + "=" * 44)
    print("           KAIRO AI STARTING")
    print("=" * 44)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading local tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True
    )

    print("Loading base model onto GPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True
    ).to("cuda")

    print("Loading existing KAIRO LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True
    )

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.use_cache = True

    print("\nKAIRO is ready!")
    print("Model: Qwen2.5-0.5B-Instruct + existing KAIRO LoRA")
    print("Inference: Local GPU")
    print("Math verification: Enabled")
    print("Prime program template: Enabled")
    print("Common Tanglish answer templates: Enabled")
    print("Python syntax validation: Enabled")
    print("=" * 44)
    print("Commands: exit | clear\n")

    return tokenizer, model


# =====================================================
# LANGUAGE RETRY
# =====================================================

def generate_answer(tokenizer, model, question, history, force_tanglish=False):
    system_prompt = SYSTEM_PROMPT

    if force_tanglish:
        system_prompt += """
STRICT LANGUAGE REQUIREMENT:
The entire answer must be natural Tamil written in English letters.
Do not use Hindi, Urdu, or English sentences.
Use English only for technical terms, names, and code.
If you cannot explain a technical term in Tamil, keep the term
in English and explain its meaning in Tanglish.
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()


# =====================================================
# RESPONSE ROUTER
# =====================================================

def ask_kairo(tokenizer, model, question, history):
    # 1. Deterministic arithmetic.
    math_answer = verify_math(question)

    if math_answer is not None:
        return math_answer

    # 2. Known programming request.
    code_answer = handle_known_programming_request(question)

    if code_answer is not None:
        validation = validate_python_code(code_answer)
        if validation:
            code_answer += "\n\nCode validation:\n" + validation
        return code_answer

    # 3. Common Tanglish question template.
    if asks_for_tanglish(question):
        template_answer = handle_known_tanglish_question(question)

        if template_answer is not None:
            return template_answer

    # 4. General model response.
    wants_tanglish = asks_for_tanglish(question)

    answer = generate_answer(
        tokenizer, model, question, history,
        force_tanglish=wants_tanglish
    )

    if not answer:
        answer = "Clear answer generate aagala. Question-ai vera maadhiri kelu."

    # Retry once with strict Tanglish instructions if the answer appears
    # to contain common Hindi words. This is a heuristic, not a perfect
    # language detector.
    if wants_tanglish:
        hindi_markers = re.compile(
            r"\b(kya|hai|hain|karna|ko bhi|mein|aur|kaise|"
            r"ke liye|nahi|nahin|iska|ek hai)\b",
            re.IGNORECASE
        )

        if hindi_markers.search(answer):
            answer = generate_answer(
                tokenizer, model, question, history,
                force_tanglish=True
            )

    if not answer:
        answer = "Clear answer generate aagala. Question-ai vera maadhiri kelu."

    validation = validate_python_code(answer)

    if validation:
        answer += "\n\nCode validation:\n" + validation

    return answer


# =====================================================
# CHAT LOOP
# =====================================================

def main():
    try:
        tokenizer, model = load_kairo()
    except Exception as error:
        print("\nKAIRO could not start.")
        print("Error:", error)
        print("Check the local model cache and output/kairo-lora folder.")
        return

    history = []

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit"):
                print("\nKAIRO: Goodbye!")
                break

            if question.lower() == "clear":
                history.clear()
                print("KAIRO: Conversation history cleared.")
                continue

            print("\nKAIRO is thinking...\n")

            try:
                answer = ask_kairo(
                    tokenizer, model, question, history
                )
                print("KAIRO:", answer)

                history.append({
                    "role": "user",
                    "content": question
                })
                history.append({
                    "role": "assistant",
                    "content": answer
                })

                history = history[-MAX_HISTORY_MESSAGES:]

                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                print(f"\n[GPU memory allocated: {memory_mb:.0f} MB]")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("GPU memory full. Try a shorter question.")

            except Exception as error:
                print("\nAnswer error:", error)

        except KeyboardInterrupt:
            print("\n\nKAIRO stopped.")
            break


if __name__ == "__main__":
    main()

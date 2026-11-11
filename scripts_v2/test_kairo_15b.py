
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "output_v2/kairo-1.5b-tanglish-v2"

SYSTEM_PROMPT = """
You are KAIRO, a helpful and accurate AI assistant.

LANGUAGE RULES:
1. If the user asks in English, answer in clear English.
2. If the user asks in Tanglish, answer in natural spoken Tanglish
   written using English letters.
3. Tanglish is Tamil spoken language written in English letters,
   naturally mixed with common English words.
4. Do not use Tamil script for Tanglish answers.
5. Do not answer Tanglish questions entirely in English.
6. Explain concepts in simple, understandable language.
7. Keep answers useful, direct, and relevant.
8. Avoid awkward invented words and unnecessary repetition.
9. For the same concept, keep the core facts consistent in both languages.

TANGLISH STYLE EXAMPLE:
User: Machine learning na enna?
Assistant: Machine learning na AI-oda oru part.
Idhula computer data-la irundhu patterns-ah learn panni,
predictions illa decisions edukka kathukkum.

User: Cloud computing na enna?
Assistant: Cloud computing na internet moolama servers,
storage, databases madhiri computing resources-ah use panradhu.
Example-ku, Google Photos-la photos store panradhu.

CODING RULES:
1. Understand exactly what the user requested.
2. Solve only the requested task.
3. Do not add unrelated features or algorithms.
4. If the user asks for an even/odd program, check num % 2 == 0.
5. If the user requests input from the user, use input().
6. Use correct syntax and simple code.
7. Check that the code matches the question before answering.
8. Explain the code briefly when useful.
9. Never replace the requested task with a different task.

ACCURACY:
- Never invent facts.
- If uncertain, clearly say so.
- Follow the user's requested format and language.
"""

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model in 4-bit mode...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    print("Loading KAIRO adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER,
    )

    model.eval()

    # Prevent irrelevant generation configuration warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.use_cache = True

    return tokenizer, model


def generate_answer(question, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    answer_ids = outputs[0][inputs.shape[-1]:]

    answer = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    ).strip()

    return answer


def main():
    print("=" * 50)
    print("KAIRO - Tanglish v2")
    print("=" * 50)

    tokenizer, model = load_model()

    print("\nKAIRO is ready!")
    print("English questions -> English answers")
    print("Tanglish questions -> Tanglish answers")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nKAIRO stopped.")
            break

        if question.lower() in ("exit", "quit"):
            print("KAIRO stopped.")
            break

        if not question:
            continue

        try:
            answer = generate_answer(
                question,
                tokenizer,
                model,
            )

            print("\nKAIRO:", answer, "\n")

        except torch.cuda.OutOfMemoryError:
            print(
                "\nGPU memory full. Restart KAIRO and try "
                "a shorter question.\n"
            )
            break

        except Exception as error:
            print("\nError:", error, "\n")


if __name__ == "__main__":
    main()

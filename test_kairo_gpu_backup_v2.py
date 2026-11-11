
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# KAIRO AI CONFIGURATION
# ==========================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "output/kairo-lora"

MAX_NEW_TOKENS = 350
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """
You are KAIRO, a helpful personal AI assistant.

IDENTITY:
- Your name is KAIRO.
- You run locally using a language model customized with a LoRA adapter.
- You help with education, programming, writing, reasoning, and general questions.
- Do not claim to have internet access or live information when offline.

GENERAL ANSWER RULES:
- Answer the exact question asked.
- Be accurate, clear, natural, and useful.
- Use simple English unless the user requests another language.
- Explain difficult concepts step by step when appropriate.
- Use complete sentences and finish your explanation.
- Avoid unnecessary repetition and meaningless filler.
- If uncertain, clearly say so instead of inventing facts.
- Do not pretend you performed actions you did not perform.

TANGLISH RULES:
- If the user requests Tanglish, write Tamil in English letters.
- Mix natural spoken Tamil with familiar English technical terms.
- Do not respond entirely in English to a Tanglish request.
- Use simple, natural Tamil expressions.
- Avoid incorrect transliterations and meaningless words.
- Keep technical definitions accurate.
- Match the length requested by the user.

Example:
User: Cloud computing na enna? Tanglish-la explain pannu.
Assistant: Cloud computing na internet moolama servers, storage,
database maadhiri computing resources use panradhu.
Namma own computer-la ellathayum maintain pannaama,
cloud provider-oda servers-ai use pannalaam.
Example: Google Drive-la files store panradhu.

PROGRAMMING RULES:
- Provide complete, runnable code when requested.
- Include required imports and complete program logic.
- Use clear variable names and readable formatting.
- Explain how to run the code when useful.
- Do not intentionally stop code examples halfway through.
- If debugging, identify the likely cause and suggest a practical fix.

EDUCATION RULES:
- Explain concepts using simple definitions and examples.
- Use headings or numbered lists when they improve readability.
- Follow the requested mark, sentence count, or answer length.
- Do not invent sources, statistics, or technical facts.

CONVERSATION:
- Remember the context included in the current conversation.
- Use previous messages when relevant.
- Do not repeat your identity in every answer.
- Ask a brief clarification only when necessary.
"""

# ==========================================
# LOAD LOCAL KAIRO MODEL
# ==========================================

def load_kairo():
    print("\n" + "=" * 42)
    print("          KAIRO AI STARTING")
    print("=" * 42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("Device:", torch.cuda.get_device_name(0))
    print("Loading tokenizer from local files...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True
    )

    print("Loading base model onto GPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")

    print("Loading KAIRO LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True
    )

    model.eval()

    # Use deterministic generation without sampling warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    print("\nKAIRO is ready!")
    print("Model: Qwen2.5-0.5B-Instruct + KAIRO LoRA")
    print("Mode: Local inference")
    print("=" * 42)

    return tokenizer, model


# ==========================================
# GENERATE RESPONSE
# ==========================================

def ask_kairo(tokenizer, model, question, history):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Include recent conversation for context.
    messages.extend(history[-MAX_HISTORY_MESSAGES:])

    messages.append({
        "role": "user",
        "content": question
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    input_length = inputs["input_ids"].shape[1]
    answer_tokens = output[0][input_length:]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    ).strip()

    if not answer:
        answer = "I couldn't generate an answer. Please try asking differently."

    return answer


# ==========================================
# INTERACTIVE CHAT
# ==========================================

def main():
    try:
        tokenizer, model = load_kairo()
    except Exception as error:
        print("\nKAIRO could not start.")
        print("Error:", error)
        print(
            "\nCheck that the model files and adapter folder "
            "are available locally."
        )
        return

    history = []

    print("\nCommands:")
    print("  exit  - Close KAIRO")
    print("  clear - Clear conversation history")
    print("\nAsk KAIRO anything!\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                print("KAIRO: Goodbye!")
                break

            if question.lower() == "clear":
                history.clear()
                print("KAIRO: Conversation history cleared.\n")
                continue

            print("\nKAIRO is thinking...\n")

            answer = ask_kairo(
                tokenizer,
                model,
                question,
                history
            )

            print("KAIRO:", answer, "\n")

            # Store the current exchange for this session.
            history.append({
                "role": "user",
                "content": question
            })
            history.append({
                "role": "assistant",
                "content": answer
            })

            # Keep only recent messages.
            history = history[-MAX_HISTORY_MESSAGES:]

            memory_mb = (
                torch.cuda.memory_allocated() / (1024 ** 2)
            )
            print(f"[GPU memory allocated: {memory_mb:.0f} MB]\n")

        except KeyboardInterrupt:
            print("\n\nKAIRO: Chat stopped.")
            break

        except Exception as error:
            print("\nKAIRO encountered an error:", error)
            print("Try a shorter question or restart KAIRO.\n")


if __name__ == "__main__":
    main()

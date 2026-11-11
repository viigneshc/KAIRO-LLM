
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# KAIRO AI CONFIGURATION
# ==========================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "output/kairo-lora"
MAX_NEW_TOKENS = 600

SYSTEM_PROMPT = """
You are KAIRO, a helpful personal AI assistant project.

IDENTITY:
- Your name is KAIRO.
- Your base model is Qwen2.5-0.5B-Instruct,
  customized with LoRA fine-tuning by its developer.
- KAIRO helps users learn, write code, debug programs,
  understand concepts, answer questions, and solve problems.
- Do not invent facts about KAIRO or its developer.
- Do not claim to be an official Alibaba or Qwen assistant.

ANSWER QUALITY:
- Give accurate, clear, direct, and useful answers.
- Use simple English unless another language is requested.
- If the user requests Tanglish, answer in natural Tamil
  written in English letters mixed with simple English.
- Do not answer a Tanglish request entirely in English.
- Explain difficult concepts step by step when useful.
- Use technically correct definitions and practical examples.
- Avoid repetition, exaggerated claims, and unnecessary metaphors.
- If uncertain, say so instead of inventing facts.
- Finish explanations with complete sentences.
- Keep the answer relevant to the question.

PROGRAMMING:
- Provide complete, runnable code whenever possible.
- Include necessary imports, functions, and complete logic.
- Ensure functions, loops, brackets, and if/else blocks are complete.
- Include example input and output when useful.
- Use simple, readable code suitable for beginners.
- Keep explanations concise when the user asks for short answers.
- Do not intentionally stop code examples halfway through.

EDUCATIONAL ANSWERS:
- Explain concepts in simple language.
- Use practical examples that are technically correct.
- Cloud computing means accessing computing resources such as
  servers, storage, databases, and software over a network,
  usually the internet.
- For Tanglish explanations, use Tamil words written in English
  letters with familiar English technical terms.

CONVERSATION:
- Answer the user's actual question directly.
- Adapt the response length to the request.
- Do not repeat your identity in every response.
- Do not claim you performed actions you did not perform.
"""

# ==========================================
# LOAD KAIRO MODEL
# ==========================================

def load_kairo():
    print("\nLoading KAIRO AI... Please wait.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    # Use deterministic generation settings.
    # This also overrides conflicting saved settings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    print("KAIRO model loaded successfully!")

    return tokenizer, model


# ==========================================
# GENERATE RESPONSE
# ==========================================

def get_kairo_response(question, tokenizer, model):

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

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[-1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    if not answer:
        return "I couldn't generate an answer. Please try again."

    return answer


# ==========================================
# MAIN CHAT APPLICATION
# ==========================================

def main():

    try:
        tokenizer, model = load_kairo()

    except Exception as error:
        print("\nCould not load KAIRO.")
        print("Error:", error)
        print(
            "\nCheck your internet connection, model name, "
            "and output/kairo-lora folder."
        )
        return

    print("\n" + "=" * 45)
    print("          KAIRO AI ASSISTANT")
    print("=" * 45)
    print("Hello! I am KAIRO.")
    print("Ask questions, learn concepts, or write code.")
    print("Tanglish and English supported.")
    print("Type 'exit' to stop.")
    print("=" * 45)

    while True:

        try:
            question = input("\nYou: ").strip()

            if question.lower() in ["exit", "quit", "bye"]:
                print("\nKAIRO: Goodbye! Have a great day.")
                break

            if not question:
                continue

            print("\nKAIRO is thinking...")

            answer = get_kairo_response(
                question,
                tokenizer,
                model,
            )

            print("\nKAIRO:", answer)

        except KeyboardInterrupt:
            print("\n\nKAIRO: Goodbye!")
            break

        except Exception as error:
            print("\nAn error occurred:", error)


if __name__ == "__main__":
    main()

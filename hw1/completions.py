from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split="test")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")


def generate_text(prompt):
    outputs = model.generate(
        tokenizer(prompt, return_tensors="pt")["input_ids"],
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.2,
        attention_mask=tokenizer(prompt, return_tensors="pt")["attention_mask"],
        top_p=None)
    
    decoded_output = tokenizer.decode(outputs[0])
    end_patterns = ["\ndef", "\nclass", "\nif", "\nprint"]
    for end_pattern in end_patterns:
        if end_pattern in decoded_output:
            decoded_output = decoded_output.split(end_pattern)[0]
            break
    return decoded_output

def generate_and_save_completions():
    for i, problem in enumerate(ds):
        for _ in range(20):
            completion = generate_text(problem['prompt'])
            if not os.path.exists("completions"):
                os.makedirs("completions")
            with open(f"completions/completion_{i}.json", "w") as file:
                json.dump(completion, file, indent=4)

generate_and_save_completions()

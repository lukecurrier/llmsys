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

def generate_and_save_completions(gens_per_prompt=20):
    if not os.path.exists("completions"):
                os.makedirs("completions")
                
    for i, problem in enumerate(ds):
        if not os.path.exists(f"completions/{i}"):
            os.makedirs(f"completions/{i}")
        for j in range(gens_per_prompt):
            print(i)
            print(problem['prompt'])
            completion = generate_text(problem['prompt'])
            with open(f"completions/{i}/completion_{j}.json", "w") as file:
                json.dump(completion, file, indent=4)

generate_and_save_completions()

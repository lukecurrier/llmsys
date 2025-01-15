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
    prompt_length = len(prompt)
    generated_text = decoded_output[prompt_length:]
    
    end_patterns = ["\ndef", "\nclass", "\nif", "\nprint"]
    for end_pattern in end_patterns:
        if end_pattern in generated_text:
            generated_text = generated_text.split(end_pattern)[0]
            break
    return prompt + generated_text

def generate_and_save_completions(gens_per_prompt=3):
    completions = []
    for i, problem in enumerate(ds):
        for j in range(gens_per_prompt):
            completion = generate_text(problem['prompt'])
            completions.append(completion)
        if not os.path.exists("completions"):
            os.makedirs("completions")
        with open(f"completions/completion_{i}.json", "w") as file:
            json.dump({
                'Prompt': problem['prompt'],
                'Completions': completions,
                'Tests': problem['tests']
            }, file, indent=4)
        completions = []

generate_and_save_completions()

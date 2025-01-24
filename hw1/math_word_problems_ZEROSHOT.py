import torch
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import re
from tqdm.auto import tqdm

test_data = datasets.load_dataset("nuprl/engineering-llm-systems", "math_word_problems", split="test")
MODEL = "/scratch/bchk/aguha/models/llama3p1_8b_base"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
).to(device=DEVICE)

def llama3(prompt, temperature=0, max_new_tokens=20, stop=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False
        generate_kwargs["temperature"] = None
    with torch.no_grad():
        example_outputs = model.generate(**inputs, **generate_kwargs, top_p=None)
    outputs = tokenizer.decode(example_outputs[0, inputs["input_ids"].shape[1]:])
    if stop is not None:
        outputs = outputs.split(stop)[0]
    return outputs

def zero_shot(problem, show_result):
    prompt = problem["question"] + "\nAnswer:"
    completion = llama3(prompt, temperature=0, max_new_tokens=5, stop="\n")
    expected = problem["answer"]
    try:
        completion_number = int(re.sub(r'[^0-9]', '', completion.strip()))
    except ValueError:
        return False
    if show_result:
        print(problem['question'])
        print(completion_number)
        print(expected)
        print(completion_number == expected)
    return completion_number == expected

zero_shot_successes = [ ]
total_problems = 0
for problem in tqdm(test_data):
    total_problems = total_problems + 1
    if zero_shot(problem, False) == True:
        zero_shot_successes.append(problem)
        
print(f"Zero-shot Success Rate: {len(zero_shot_successes)/total_problems}")

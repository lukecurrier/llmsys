import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
api_key = os.getenv("HUGGINGFACE_TOKEN")

model_name = "google/gemma-3-27b-it" 
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key)

def summarize_markdown(markdown_path):
    with open(markdown_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    prompt = f"""Please summarize the following markdown content:
    
{text}

Summary:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
    
    summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    markdown_filename = input("Enter the name of the markdown file: ")
    markdown_path = os.path.join("conversions", f"{markdown_filename}.md")
    
    if not os.path.exists(markdown_path):
        print(f"File not found: {markdown_path}")
    else:
        summary = summarize_markdown(markdown_path)
        print(summary)
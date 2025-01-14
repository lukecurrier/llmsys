# This program executes the completed code prompts held in /completions 
# against the dataset's test suite. It prints a success rate. 

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split="test")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")


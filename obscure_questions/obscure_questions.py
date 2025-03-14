import math
from functools import lru_cache
import os
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import heapq
from typing import List

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

CUSTOM_STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'of', 'at', 'by',
    'for', 'to', 'in', 'on', 'is', 'it', 'this', 'that', 'these', 'those', 'as',
    'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did',
    'from', 'not', 'can', 'will', 'would', 'should', 'could', 'i', 'you', 'he', 
    'she', 'we', 'they', 'them', 'his', 'her', 'its', 'our', 'your', 'their',
    'about', 'above', 'below', 'between', 'each', 'few', 'more', 'most', 'some', 
    'such', 'only', 'own', 'same', 'than', 'then', 'there', 'when', 'where', 
    'while', 'how', 'all', 'any', 'both', 'each', 'few', 'here', 'over', 'under',
    'again', 'further', 'once', 'before', 'after', 'during', 'until', 'without',
    'within', 'per', 'via', 'now', 'get', 'go', 'let', 'us', 'use', 'used', 'using',
    'make', 'made', 'know', 'known', 'take', 'taken', 'see', 'seen', 'look', 'looked',
    'come', 'came', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'help', 'need',
    'think', 'try', 'change', 'changed'
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
base_url = "https://nerc.guha-anderson.com/v1"
api_key = "ravuri.n@northeastern.edu:81592"
client = OpenAI(base_url=base_url, api_key=api_key)

neu_wiki = load_dataset("nuprl/engineering-llm-systems", name="wikipedia-northeastern-university", split="test")
obscure_questions = load_dataset("nuprl/engineering-llm-systems", name="obscure_questions", split="test")

def preprocess_text(text: str):
    tokens = text.lower().split()  # Simple whitespace tokenization
    return [token.strip(".,!?()[]{}\"'") for token in tokens if token not in CUSTOM_STOP_WORDS]


@lru_cache(maxsize=10000)
def term_frequency(document: str, term: str):
    c = document.count(term)
    return 0 if c == 0 else 1 + math.log(c)
    # return document.count(term)

@lru_cache(maxsize=None)
def inverse_document_frequency(term: str):
    num_docs_with_term = sum(1 for item in neu_wiki if term in item["text"])
    return math.log((1 + len(neu_wiki)) / (1 + num_docs_with_term)) + 1

def compute_tf_idf_vector_unnormalized(terms, document: str):
    return [ term_frequency(document, term) * inverse_document_frequency(term) for term in terms ]

def compute_tf_idf_vector(terms, document: str):
    vec = compute_tf_idf_vector_unnormalized(terms, document)
    return vec

def compute_cosine_similarity(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm == 0 or vec2_norm == 0:
        return 0
    
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

def rank_by_tf_idf(query: str):
    query_vec = compute_tf_idf_vector(query.split(), query)
    return sorted(neu_wiki, key=lambda x: compute_cosine_similarity(query_vec, compute_tf_idf_vector(query.split(), x["text"])), reverse=True)

def find_top_n_documents(n, prompt):
    with torch.no_grad():
        query_docs = rank_by_tf_idf(prompt)
        #print("\n" + prompt + "\n-")
        #for doc in query_docs[:2]: 
        #   print(doc['title'])
        #print("\n")
        top_docs = []  # Min-heap to store top n documents
        query_vec = model(**tokenizer(prompt, return_tensors="pt")).last_hidden_state[0, 0]
        
        for doc in query_docs[:2]:
            #print("Examining " + doc['title'])
            doc_vec = model(**tokenizer(doc["text"], return_tensors="pt", truncation=True)).last_hidden_state[0, 0]
            cosine_sim = compute_cosine_similarity(query_vec.numpy(), doc_vec.numpy())
            #print(f"Similarity: {cosine_sim:.4f}\n")
            
            # Maintain a min-heap of top n documents
            if len(top_docs) < n:
                heapq.heappush(top_docs, (cosine_sim, doc))
            else:
                heapq.heappushpop(top_docs, (cosine_sim, doc))
        #print("\n")

        # Sort in descending order based on similarity
        sorted_top_docs = sorted(top_docs, key=lambda x: x[0], reverse=True)

        #print(f"Top documents for question: {prompt}")
        #for sim, doc in sorted_top_docs:
        #    print(f"Similarity: {sim:.4f}")
        #    print(doc['title'])
        #    print(doc['url'])
        #print("\n" + "-" * 50 + "\n")
        return [doc for _, doc in sorted_top_docs]


def answer_query(question: str, choices: List[str], documents: List[str]) -> str:
    """
    Answers a multiple choice question using retrieval augmented generation.

    `question` is the text of the question. `choices` is the list of choices
     with leading letters. For example:

     ```
     ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"]
     ```

     `documents` is the list of documents to use for retrieval augmented
     generation.

     The result should be the just the letter of the correct choice, e.g.,
     `"A"` but not `"A."` and not `"A. Choice 1"`.
     """

    SYSTEM_PROMPT = """You are a question-answering assistant tasked with answering multiple-choice questions with the help of relevant documents. 
    You will be provided with a question, a list of choices, and a list of relevant documents. 
    Your task is to use the provided documents to find the correct answer and return it as a letter. 
    Format your answer as simply as possible, returning ONLY the letter of the correct answer. 
    For example, if the correct answer is 'Choice 1', return 'A' but not `A.` and not `A. Choice 1`."""

    USER_PROMPT = f"""Answer the following multiple-choice question: {question}\nYou have {len(documents)} 
    relevant documents to use to help you find the correct answer:\n{'\n'.join([f"{doc['title']} ({doc['text']})" for doc in documents])}
    \nThe potential answers are:\n{'\n'.join([f"{choice}" for choice in choices])}\n\nYour answer is:\n"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        model="llama3p1-8b-instruct",
        temperature=0,
        max_tokens=5
    )
    return response.choices[0].message.content[0]


if __name__ == "__main__":
    obscure_questions = obscure_questions.select(range(10))
    total_questions = 0
    correct_answers = 0
    for obsquestion in obscure_questions:
        total_questions = total_questions + 1
        question = obsquestion['prompt']
        choices = obsquestion['choices']

        print(f"Question: {question}")

        documents = find_top_n_documents(2, question)
        for doc in documents:
            print(f"{doc['title']} ({doc['url']})")

        answer = answer_query(question, choices, documents)
        print(f"Choices: {choices}")
        print(f"Answer: {answer}")
        print(f"Correct Answer: {obsquestion['correct_answer']}")
        print("----------------------------------\n")
        
        if obsquestion['correct_answer'] == answer:
            correct_answers = correct_answers + 1
    print(f"Total Accuracy: {(correct_answers/total_questions) * 100}%")

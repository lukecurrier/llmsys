import math
from functools import lru_cache
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
#from openai import OpenAI
import heapq
from typing import List

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

neu_wiki = load_dataset("nuprl/engineering-llm-systems", name="wikipedia-northeastern-university", split="test")
obscure_questions = load_dataset("nuprl/engineering-llm-systems", name="obscure_questions", split="test")

def term_frequency(document: str, term: str):
    c = document.count(term)
    return 0 if c == 0 else 1 + math.log(c)
    # return document.count(term)

@lru_cache(maxsize=None)
def inverse_document_frequency(term: str):
    num_docs_with_term = sum(1 for item in neu_wiki if term in item["text"])
    return math.log(len(neu_wiki) / (1 + num_docs_with_term))

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

def find_top_n_documents(n):
    with torch.no_grad():
        for question in obscure_questions:
            query_docs = rank_by_tf_idf(question['prompt'])
            top_docs = []  # Min-heap to store top n documents
            query_vec = model(**tokenizer(question['prompt'], return_tensors="pt")).last_hidden_state[0, 0]
            
            for doc in query_docs[:2]:
                doc_vec = model(**tokenizer(doc["text"], return_tensors="pt", truncation=True)).last_hidden_state[0, 0]
                cosine_sim = compute_cosine_similarity(query_vec.numpy(), doc_vec.numpy())
                
                # Maintain a min-heap of top n documents
                if len(top_docs) < n:
                    heapq.heappush(top_docs, (cosine_sim, doc))
                else:
                    heapq.heappushpop(top_docs, (cosine_sim, doc))

            # Sort in descending order based on similarity
            sorted_top_docs = sorted(top_docs, key=lambda x: x[0], reverse=True)

            print(f"Top documents for question: {question['prompt']}")
            for sim, doc in sorted_top_docs:
                print(f"Similarity: {sim:.4f}")
                print(doc['title'])
                print(doc['url'])
            print("\n" + "-" * 50 + "\n")


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

find_top_n_documents(2)

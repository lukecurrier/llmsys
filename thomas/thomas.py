from openai import OpenAI
from datasets import load_dataset
import datetime
from typing import List, Optional

flights = load_dataset("nuprl/engineering-llm-systems", name="flights", split="train")

class Flight:
    def __init__(self, flight_id: int):
        self.flight_info = flights[flight_id]

def find_flights(origin: str, destination: str, date: datetime.date) -> List[Flight]:
    formatted_date = date.strftime("%Y-%m-%d")
    return [Flight(flight["id"]) for flight in flights if flight["origin"] == origin and flight["destination"] == destination and flight["date"] == formatted_date]

def book_flight(flight_id: int) -> Optional[int]:
    if flights[flight_id]["available_seats"] > 0:
        return flight_id
    else: 
        return None


def run_chat(api_key: str):
    base_url = "https://nerc.guha-anderson.com/v1"
    prompt_prefix = """You are an AI travel agent. Talk to the user as helpfully and briefly as possible, 
    answering their questions when possible but politely denying any query unrelated to travel.
    
    User input:"""

    client = OpenAI(base_url=base_url, api_key=api_key)

    while True: 
        user_input = input("User (blank to quit):")

        if user_input == "":
            print("[]")
            break

        # chatbot logic here
        resp = client.chat.completions.create(
            messages = [{ 
                "role": "user", "content": prompt_prefix + user_input 
            }],
            model = "llama3p1-8b-instruct",
            temperature=0)
        print(resp.choices[0].message.content)

run_chat(api_key = "currier.l@northeastern.edu:19284")
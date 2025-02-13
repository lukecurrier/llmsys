from openai import OpenAI
from datasets import load_dataset
import datetime
import re
from typing import List, Optional

OG_FLIGHT = load_dataset("nuprl/engineering-llm-systems", name="flights", split="train")

class Flight:
    def __init__(self, flight_id: int):
        self.flight_info = OG_FLIGHT[flight_id]
    
def find_flights(origin: str, destination: str, date: datetime.date) -> List[Flight]:
    formatted_date = date.strftime("%Y-%m-%d")
    return [Flight(flight["id"]) for flight in OG_FLIGHT if flight["origin"] == origin and flight["destination"] == destination and flight["date"] == formatted_date]

def book_flight(flight_id: int) -> Optional[int]:
    if OG_FLIGHT[flight_id]["available_seats"] > 0:
        return flight_id
    else: 
        return None


def run_chat(api_key: str):
    base_url = "https://nerc.guha-anderson.com/v1"
    
    SYSTEM_PROMPT = """
    
    We have defined a function called

    def find_flights(origin: str, destination: str, departure_date: datetime.date) -> list[dict]:

    It takes the origin and destination airport codes. And produces a list of dictionaries
    containing flight information. Stuff like this:

    { 'id': 407,
    'date': datetime.date(2023, 1, 6),
    'airline': 'Delta',
    'flight_number': 'DL9926',
    'origin': 'ORD',
    'destination': 'JFK',
    'departure_time': datetime.time(20, 55),
    'arrival_time': datetime.time(22, 55),
    'available_seats': 172}
    
    When a user asks about the available flights, use the find_flights function to find the list of flights matching the specified
    origin, destination and date
    
    After showing the available flights, if a user chooses one, use the book_flight function as defined below to book the chosen flight:
    
    def book_flight(flight_id: int) -> Optional[int]:
    
    If the flight booking is succesful, the functino will return the flight id, else it will return None. Inform the user if the booking was
    succesful or not. 
    
    Please return python code with all the above actions.
    
    I will run your response code in an environment where find_flights and book_flight are already defined
    and datetime is already imported. Please do not name any variables flights and do not redefine or mention the definition for find_flights or book_flight.
    Assume the main function is already defined, do just provide the code inside the main() function.
    """

    client = OpenAI(base_url=base_url, api_key=api_key)

    while True: 
        user_input = input("User (blank to quit):")

        if user_input == "":
            print("[]")
            break

        # chatbot logic here
        resp = client.chat.completions.create(
            messages = [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": user_input}
                ],
            model = "llama3p1-8b-instruct",
            temperature=0)
        extracted_code = re.search(r'```python(.*?)```', resp.choices[0].message.content, re.DOTALL)
        if extracted_code:
            extracted_code = extracted_code.group(1)
            try:
                print(exec(extracted_code))
            except Exception as e:
                print(e)
        else:
            print(resp.choices[0].message.content)
            
        
run_chat(api_key = "ravuri.n@northeastern.edu:81592")

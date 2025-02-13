from openai import OpenAI
from datasets import load_dataset
import datetime
import re
from typing import List, Optional

OG_FLIGHT = load_dataset("nuprl/engineering-llm-systems", name="flights", split="train")

class Flight:
    def __init__(self, flight_id: int):
        self.id = OG_FLIGHT[flight_id]['id']
        self.date = OG_FLIGHT[flight_id]['date']
        self.airline = OG_FLIGHT[flight_id]['airline']
        self.flight_number = OG_FLIGHT[flight_id]['flight_number']
        self.origin = OG_FLIGHT[flight_id]['origin']
        self.destination = OG_FLIGHT[flight_id]['destination']
        self.departure_time = OG_FLIGHT[flight_id]['departure_time']
        self.arrival_time = OG_FLIGHT[flight_id]['arrival_time']
        self.available_seats = OG_FLIGHT[flight_id]['available_seats']
    
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
    
    SYSTEM_PROMPT  = """
    We have defined a function called

    def find_flights(origin: str, destination: str, departure_date: datetime.date) -> list[Flight]:

    It takes the origin and destination airport codes. And produces a list of Flight objects
    containing flight information. Each Flight object has the following fields: id, date, airline, flight_number, origin
    destination, departure_time, arrival_time, and available_seats. 
    
    I will run your response code in an environment where find_flights is already defined
    and datetime is already imported. So please do not define find_flights. Make sure your response code displays the available flights
    in a user friendly manner that only displays the flight id, departure_time, and available_sears.
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
                #print(extracted_code)
                print(exec(extracted_code))
            except Exception as e:
                print(e)
        else:
            print(resp.choices[0].message.content)
            
        
run_chat(api_key = "ravuri.n@northeastern.edu:81592")
#What flights are there from Boston to San Francisco on January 6, 2023?

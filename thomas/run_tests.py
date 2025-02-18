from openai import OpenAI
from datasets import load_dataset
import datetime
import re
from typing import List, Optional
from contextlib import redirect_stdout
import io
import sys
import argparse

OG_FLIGHT = load_dataset("nuprl/engineering-llm-systems", name="flights", split="train")
booked_flights = []

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
        booked_flights.append(flight_id)
        return flight_id
    else: 
        return None


def run_chat(api_key: str, model: str):
    base_url = "https://nerc.guha-anderson.com/v1"
    
    SYSTEM_PROMPT  = """
    You are an AI travel agent named Thomas conversing with a user. If the user requests information about flights, your task is to find and book their flights. You MUST generate code to find and book flights, if requested.
    Do not display available flights without generating code to do so.
    We have defined a function called

    def find_flights(origin: str, destination: str, departure_date: datetime.date) -> list[Flight]:

    It takes the origin and destination airport codes. And produces a list of Flight objects
    containing flight information. Each Flight object has the following fields: id, date, airline, flight_number, origin
    destination, departure_time, arrival_time, and available_seats. 
    
    We have also defined a function called

    book_flight(flight_id: int) -> Optional[int]

    It takes the flight id as input and returns the flight id if the flight has available seats. If the flight
    does not have available seats, it returns None. 
    
    I will run your response code in a dynamic environment where find_flights and book_flight is already defined
    and datetime is already imported. Do not reimport datetime. Make sure the code you produce is syntactically correct and does not have empty functions. 
    So please DO NOT redefine the find_flights or book_flight function or provide function signatures for them. Make sure your response code displays the available flights
    in a user friendly manner that only displays the flight id, departure_time, and available_seats. 
    """
    messages = [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": "What flights are there from Boston to San Francisco on January 6, 2023?"},
                { "role": "system", "content": '''```python
flights = find_flights('BOS', 'SFO', datetime.date(2023, 1, 6))
if flights:
   print("Available Flights:")
   for flight in flights:
        print(f"Flight ID: {flight.id}, Departure Time: {flight.departure_time}, Available Seats: {flight.available_seats}")
else:
     print("No flights available.")
```
Available Flights:
Flight ID: 474, Departure Time: 05:29, Available Seats: 0
Flight ID: 475, Departure Time: 18:12, Available Seats: 90
Flight ID: 476, Departure Time: 00:17, Available Seats: 0''' },
                { "role": "user", "content": "can i book the second flight"},
                { "role": "system", "content": '''```python
booking = book_flight(475)
if booking:
    print(f"Booking for flight {booking} succesful")
else:
   print(f"Flight is unfortunately at full capacity. Please choose a different flight.")''' },
                { "role": "user", "content": "What are the flights from Boston to EWR on jan 1?"},
                { "role": "system", "content": '''```python
flights = find_flights('BOS', 'EWR', datetime.date(2023, 1, 1))
if flights:
   print("Available Flights:")
   for flight in flights:
        print(f"Flight ID: {flight.id}, Departure Time: {flight.departure_time}, Available Seats: {flight.available_seats}")
    else:
        print("No flights available.")
```
Available Flights:
Flight ID: 1, Departure Time: 05:29, Available Seats: 0
Flight ID: 2, Departure Time: 18:12, Available Seats: 90
Flight ID: 3, Departure Time: 00:17, Available Seats: 0''' },
                { "role": "user", "content": "i'll do the first"},
                { "role": "system", "content": '''```python
booking = book_flight(1)
if booking:
    print(f"Booking for flight {booking} succesful")
else:
   print(f"Flight is unfortunately at full capacity. Please choose a different flight.")''' },
                ]        

    client = OpenAI(base_url=base_url, api_key=api_key)

    while True: 
        user_input = input("User (blank to quit):")

        if user_input == "":
            print(booked_flights)
            break
        
        #Append the message the user asked to the log
        messages.append({ "role": "user", "content": user_input})
        
        # chatbot logic here
        resp = client.chat.completions.create(
            messages = messages,
            #model = "llama3p1-8b-instruct",
            model = model,
            temperature=0)
        extracted_code = re.search(r'```python(.*?)```', resp.choices[0].message.content, re.DOTALL)
        if extracted_code:
            extracted_code = extracted_code.group(1)
            try:
                print(extracted_code)
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exec(extracted_code)
                print(stdout.getvalue())
                # Append the system's answer and code to the log
                messages.append({ "role": "system", "content":  stdout.getvalue()})
                #print(messages)
            except Exception as e:
                print(e)
        else:
            messages.append({ "role": "system", "content": resp.choices[0].message.content})
            print(resp.choices[0].message.content)
            

def main():
    parser = argparse.ArgumentParser(description="Thomas the Travel Agent")
    parser.add_argument('model', help="The model name to use for the chatbot")
    args = parser.parse_args()
    run_chat(api_key = "ravuri.n@northeastern.edu:81592", model=args.model)

if __name__ == "__main__":
    main()
    
#run_chat(api_key = "ravuri.n@northeastern.edu:81592")
#What flights are there from Boston to San Francisco on January 6, 2023?

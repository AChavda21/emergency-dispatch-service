# The Telephony Simulator
# This script simulates a 911 dispatch center receiving calls in real-time. 
# It generates a transcript every few seconds and sends a live HTTP POST request to your FastAPI server.

import requests
import random
import time
import string
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - FIREHOSE - %(message)s")

API_ENDPOINT = "http://127.0.0.1:8000/v1/triage/analyze"

def generate_live_call():
    locations = ["104th Street", "Jasper Ave", "Whyte Ave", "the LRT station", "Southgate Mall"]
    events = [
        "guy with a knife threatening people",
        "massive car crash, someone is trapped in the vehicle",
        "apartment on the 3rd floor is filled with thick black smoke",
        "my grandfather just collapsed and is clutching his chest",
        "bunch of teenagers smashing car windows in the parking lot"
    ]
    caller_id = f"CALL-{''.join(random.choices(string.ascii_uppercase + string.digits, k=6))}"
    transcript = f"Emergency at {random.choice(locations)}, there is a {random.choice(events)}. Send units!"
    
    return {"caller_id": caller_id, "transcript": transcript}

def run_firehose():
    logging.info("Starting live CAD stream simulator. Pushing to API...")
    
    while True:
        payload = generate_live_call()
        logging.info(f"Incoming Call: {payload['caller_id']} - '{payload['transcript']}'")
        
        try:
            response = requests.post(API_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                alert = "🚨 HIGH SEVERITY" if data['severity_flag'] else "ℹ️ STANDARD"
                review = " 🧑 NEEDS HUMAN REVIEW" if data.get('needs_human_review') else ""
                logging.info(f"API ROUTING DECISION: {data['assigned_department']} [{alert}] "
                             f"(confidence {data.get('confidence', 0):.0%}){review}\n")
            else:
                logging.error(f"API Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logging.error("API Unreachable. Is the FastAPI server running?")
            
        # Simulate the random time delay between 911 calls
        time.sleep(random.uniform(2.0, 5.0))

if __name__ == "__main__":
    run_firehose()
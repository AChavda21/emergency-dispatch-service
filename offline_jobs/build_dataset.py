# The Data Warehouse Script
# This generates the offline training data. In a real environment, this might pull from a Snowflake or Azure SQL database.

import sys
import os
import pandas as pd
import random
import logging

# Append project root to path so we can import the shared config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import TRAINING_CORPUS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LOCATIONS = [
    "on Anthony Henday Drive", "at the West Edmonton Mall", "in a residential garage", "outside the downtown library",
    "at the industrial park", "on Jasper Ave", "at the LRT station", "on Whyte Ave", "near 104th Street",
    "at Southgate Mall", "in the parking lot", "at my apartment building", "on the Yellowhead Trail", "behind the school",
]

# Verb phrases: "Emergency {location}, someone {scenario}."
SCENARIOS = {
    0: ["collapsed and is not breathing", "is having severe chest pains", "fell from a ladder and is unconscious",
        "is choking and turning blue", "is having a seizure on the floor", "passed out and won't wake up",
        "is bleeding heavily from a head wound", "is having a bad allergic reaction and can't breathe"],
    1: ["is engulfed in flames", "has heavy black smoke pouring out", "caught fire in the kitchen spreading to the roof",
        "has a ruptured gas line that ignited", "set a dumpster on fire next to the building",
        "has flames coming out of the windows", "left a car burning in the lot"],
    2: ["is kicking my door down with a weapon", "just shot someone in the alley", "is aggressively assaulting a pedestrian",
        "is waving a knife at the cashier", "is pointing a gun at people", "just stabbed a man outside the bar",
        "is beating someone with a bat"],
    3: ["was just spray painted by kids", "had its windows smashed in overnight", "was broken into and my bike is missing",
        "is being vandalized right now", "stole a package off my porch", "is shoplifting from the store",
        "keyed all the cars on the street"],
    4: ["has a massive chemical spill leaking yellow gas", "collapsed trapping workers in the trench",
        "is covered in hazardous radioactive material", "has a derailed train car leaking acid",
        "is trapped inside a crashed car", "fell through the ice on the river", "is pinned under debris after a collapse"],
}

# Noun phrases: "Emergency {location}, there is a {incident}."
INCIDENTS = {
    0: ["man who collapsed and is clutching his chest", "woman having a seizure", "child who is choking",
        "elderly lady who fell and is unresponsive", "person who overdosed and is barely breathing",
        "diabetic man who passed out"],
    1: ["house fire spreading to the roof", "apartment filled with thick black smoke", "car on fire",
        "kitchen fire that is out of control", "brush fire moving toward the houses", "electrical fire in the basement"],
    2: ["man with a knife threatening people", "shooting in progress", "guy with a gun robbing the store",
        "violent fight with someone being stabbed", "domestic assault happening next door", "armed robbery at the bank"],
    3: ["group of teenagers smashing car windows", "break-in at an empty house", "shoplifter running from the store",
        "bunch of kids spray painting the walls", "stolen bike being sold on the corner", "car that was broken into overnight"],
    4: ["massive car crash with someone trapped in the vehicle", "chemical leak with toxic fumes",
        "worker trapped in a collapsed trench", "multi-vehicle pileup with people pinned inside",
        "gas tanker spilling fuel on the road", "person stuck in a flooded storm drain"],
}

URGENCIES = ["Send someone immediately!", "Please hurry!", "We are evacuating now.", "I'm hiding, please be quiet.",
             "It looks really bad.", "Send help now!", "Send units!", "I don't know what to do."]

TEMPLATES = [
    lambda c: f"Emergency {random.choice(LOCATIONS)}, someone {random.choice(SCENARIOS[c])}. {random.choice(URGENCIES)}",
    lambda c: f"Emergency {random.choice(LOCATIONS)}, there is a {random.choice(INCIDENTS[c])}. {random.choice(URGENCIES)}",
    lambda c: f"{random.choice(URGENCIES)} There's a {random.choice(INCIDENTS[c])} {random.choice(LOCATIONS)}.",
]


def build_training_corpus(num_records=2000, output_path=TRAINING_CORPUS_PATH, seed=42):
    random.seed(seed)
    data = []
    for _ in range(num_records):
        category = random.randint(0, 4)
        text = random.choice(TEMPLATES)(category)
        data.append({"label": category, "transcript": text})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logging.info(f"Training corpus compiled: {num_records} records saved to {output_path}")


if __name__ == "__main__":
    build_training_corpus()

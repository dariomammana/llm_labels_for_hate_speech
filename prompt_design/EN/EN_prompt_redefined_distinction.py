import anthropic
import csv
import time
import pandas as pd

comments = pd.read_csv("Data/Sample150_EN.csv")["text"].tolist()

# --- Personas (Modified Text) ---
# Base definitions from Kralj Novak 2022
SCALE_DEFINITIONS = """
3, VIOLENT, meaning it is directed at a target with violent content. it can also be 
explicit physical violence or implicit reference of denial or glorification of historical violence.
2, OFFENSIVE, targeted at individuals or groups including offensive generalization, contempt, dehumanization, indirect
offensive remarks, but not violent.
1, INAPPROPRIATE, if the comment has no target but contains terms that are obscene or vulgar, but not violent nor offensive.
0, APPROPRIATE, if the comment is not directed at anyone and is not any of the above.

Respond with ONLY a single digit: 0, 1, 2, or 3. Nothing else."""

# Persona 1: Simulates a highly tolerant internet user who ignores mild insults
SYSTEM_PROMPT_1 = """You are a casual social media user reviewing YouTube comments.
You have a high tolerance for targeted insults but you are easily offended by 
crude, vulgar or obscene language even when it is not directed at anyone in particular.
A comment does not need to target anyone to be inappropriate — general vulgarity 
or obscenity is enough to classify it as 1.""" + SCALE_DEFINITIONS

SYSTEM_PROMPT_2 = """You are a strict content moderator reviewing YouTube comments.
You apply zero tolerance for any form of crude, vulgar or obscene language regardless 
of whether it targets anyone specifically.
A comment does not need to target anyone to be inappropriate — general vulgarity 
or obscenity alone is enough to classify it as 1.""" + SCALE_DEFINITIONS
# Map runs to specific prompts
SYSTEM_PROMPTS = [SYSTEM_PROMPT_1, SYSTEM_PROMPT_2]

# --- SET TEMPERATURE HERE ---
# Force maximum variance between the two personas
TEMPERATURE = 1.0  

# --- Client ---
client = anthropic.Anthropic()

# --- Output file ---
OUTPUT_FILE = f"EN_prompt_redefined_distinction.csv"

# Write header
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "comment", "label_run1", "label_run2"])

# --- Label twice per comment using alternating personas ---
for i, comment in enumerate(comments):
    labels = []

    for run in range(2):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPTS[run],  # <--- Dynamically switches prompt per run
                messages=[{"role": "user", "content": comment}]
            )
            label = response.content[0].text.strip()

            if label not in ["0", "1", "2", "3"]:
                print(f"[WARNING] Unexpected output for comment {i}, run {run+1}: '{label}'")

        except Exception as e:
            label = "ERROR"
            print(f"[ERROR] Comment {i}, run {run+1} failed: {e}")

        labels.append(label)
        time.sleep(0.3)

    # Append both labels to CSV
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([i, comment, labels[0], labels[1]])

    print(f"[{i+1}/{len(comments)}] Run1 (Literal): {labels[0]} | Run2 (Sensitive): {labels[1]} | {comment[:60]}")

print(f"\nDone. Results saved to {OUTPUT_FILE}")

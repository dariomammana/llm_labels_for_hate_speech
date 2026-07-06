import anthropic
import json
import time
import pandas as pd

# --- Config ---
INPUT_FILE = "../Data/Unique_EN.csv"
REGISTRY_FILE = "batch_registry_EN.json"
BATCH_SIZE = 10000
TEMPERATURE = 1.0

# --- Prompts ---
SCALE_DEFINITIONS = """
3, VIOLENT, meaning it is directed at a target with violent content. it can also be 
explicit physical violence or implicit reference of denial or glorification of historical violence.
2, OFFENSIVE, targeted at individuals or groups including offensive generalization, contempt, dehumanization, indirect
offensive remarks, but not violent.
1, INAPPROPRIATE, if the comment has no target but contains terms that are obscene or vulgar, but not violent nor offensive.
0, APPROPRIATE, if the comment is not directed at anyone and is not any of the above.

Respond with ONLY a single digit: 0, 1, 2, or 3. Nothing else."""

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

SYSTEM_PROMPTS = {
    "run1": SYSTEM_PROMPT_1,
    "run2": SYSTEM_PROMPT_2,
}

# --- Load comments ---
print(f"Loading comments from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
comments = df["text"].tolist()
print(f"Total comments: {len(comments)}")

# --- Split into chunks ---
chunks = [comments[i:i+BATCH_SIZE] for i in range(0, len(comments), BATCH_SIZE)]
print(f"Chunks: {len(chunks)} x up to {BATCH_SIZE} comments")
print(f"Total batches to submit: {len(chunks) * 2} (2 runs x {len(chunks)} chunks)\n")

# --- Client ---
client = anthropic.Anthropic()

# --- Submit ---
registry = []  # list of {batch_id, run_name, start_index, size}

for run_name in ["run1", "run2"]:
    print(f"=== Submitting {run_name} ===")
    for chunk_idx, chunk in enumerate(chunks):
        start_index = chunk_idx * BATCH_SIZE

        requests = []
        for i, comment in enumerate(chunk):
            original_index = start_index + i
            requests.append({
                "custom_id": f"{run_name}_{original_index}",
                "params": {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 5,
                    "temperature": TEMPERATURE,
                    "system": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPTS[run_name],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [
                        {"role": "user", "content": comment}
                    ]
                }
            })

        batch = client.messages.batches.create(requests=requests)
        print(f"  Chunk {chunk_idx+1}/{len(chunks)} | batch_id={batch.id} | {len(chunk)} requests")

        registry.append({
            "batch_id": batch.id,
            "run_name": run_name,
            "start_index": start_index,
            "size": len(chunk)
        })

        time.sleep(2)  # brief pause between submissions

# --- Save registry ---
with open(REGISTRY_FILE, "w") as f:
    json.dump(registry, f, indent=2)

print(f"\nAll batches submitted. Registry saved to {REGISTRY_FILE}")
print("Now run retrieve_results_EN.py when batches are complete (usually 1-2 hours).")

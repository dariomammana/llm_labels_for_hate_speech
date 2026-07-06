import anthropic
import json
import time
import pandas as pd

# --- Config ---
INPUT_FILE = "../Data/Unique_IT.csv"
REGISTRY_FILE = "batch_registry_IT.json"
BATCH_SIZE = 10000
TEMPERATURE = 1.0

# --- Prompts ---
SCALE_DEFINITIONS = """
3, VIOLENTO, ovvero che è rivolto a un bersaglio con contenuti violenti. Può trattarsi anche di violenza fisica esplicita o di un riferimento implicito alla negazione o glorificazione della violenza storica. 
2, OFFENSIVO, rivolto a individui o gruppi, comprese generalizzazioni offensive, disprezzo, disumanizzazione, osservazioni offensive indirette, ma non violente. 
1, INAPPROPRIATO, se il commento non ha un bersaglio ma contiene termini osceni o volgari, ma non violenti né offensivi. 
0, APPROPRIATO, se il commento non è rivolto a nessuno e non rientra in nessuno dei casi precedenti.

Rispondi solamente con una cifra singola: 0, 1, 2 o 3. Nient'altro."""

SYSTEM_PROMPT_1 = """Sei un utente casuale di social media che valuta i commenti di YouTube.
Hai un'alta tolleranza per gli insulti mirati, ma ti offendi facilmente per il linguaggio crudo, volgare o osceno, anche quando non è diretto a nessuno in particolare.
Un commento non deve per forza colpire qualcuno per essere inappropriato — la volgarità o l'oscenità generale è sufficiente per classificarlo come 1.""" + SCALE_DEFINITIONS

SYSTEM_PROMPT_2 = """Sei un moderatore di contenuti severo che valuta i commenti di YouTube.
Applichi la tolleranza zero per qualsiasi forma di linguaggio crudo, volgare o osceno, indipendentemente dal fatto che sia diretto specificamente a qualcuno.
Un commento non ha bisogno di colpire nessuno per essere inappropriato — la sola volgarità o oscenità generale è sufficiente per classificarlo come 1.""" + SCALE_DEFINITIONS

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
registry = []

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

        time.sleep(2)

# --- Save registry ---
with open(REGISTRY_FILE, "w") as f:
    json.dump(registry, f, indent=2)

print(f"\nAll batches submitted. Registry saved to {REGISTRY_FILE}")
print("Now run retrieve_results_IT.py when batches are complete (usually 1-2 hours).")

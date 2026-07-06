import anthropic
import json
import time
import pandas as pd

# --- Config ---
EXISTING_FILE = "../Data/SI_annotations.csv"
REGISTRY_FILE = "batch_registry_SI_retry.json"
INPUT_FILE = "../Data/Unique_SI.csv"
BATCH_SIZE = 10000
TEMPERATURE = 1.0

# --- Prompts ---
SCALE_DEFINITIONS = """
3, NASILNO (VIOLENT), kar pomeni, da je usmerjeno v tarčo z nasilno vsebino. 
Lahko gre tudi za eksplicitno fizično nasilje ali implicitno nanašanje na zanikanje oziroma veličanje zgodovinskega nasilja.
2, ŽALJIVO (OFFENSIVE), usmerjeno v posameznike ali skupine, vključno z žaljivim posploševanjem, preziranjem, razčlovečenjem, posrednimi žaljivimi opazkami, vendar ni nasilno.
1, NEPRIMERNO (INAPPROPRIATE), če komentar nima tarče, vendar vsebuje izraze, ki so opsceni ali vulgarni, vendar niso nasilni ali žaljivi.
0, PRIMERNO (APPROPRIATE), če komentar ni usmerjen v nikogar in ne ustreza ničemur od zgoraj navedenega.

Odgovori SAMO z eno cifro: 0, 1, 2 ali 3. Nič drugega."""

SYSTEM_PROMPT_1 = """Ste običajen uporabnik družbenih omrežij, ki ocenjuje objave na Twitterju.
Imate visoko toleranco do usmerjenih žaljivk, vendar vas hitro užali grob, vulgaren ali opolzek jezik, tudi če ni usmerjen proti komurkoli posebej.
Objavi ni treba biti usmerjena proti komurkoli, da bi bila neprimerna — splošna vulgarnost ali opolzkost je dovolj, da jo razvrstite kot 1.""" + SCALE_DEFINITIONS

SYSTEM_PROMPT_2 = """Ste strog moderator vsebin, ki ocenjuje objave na Twitterju.
Uporabljate ničelno toleranco do kakršnekoli oblike grobega, vulgarnega ali opolzkega jezika, ne glede na to, ali je usmerjen proti komurkoli posebej.
Objavi ni treba biti usmerjena proti komurkoli, da bi bila neprimerna — že sama splošna vulgarnost ali opolzkost zadostuje, da jo razvrstite kot 1.""" + SCALE_DEFINITIONS

SYSTEM_PROMPTS = {
    "run1": SYSTEM_PROMPT_1,
    "run2": SYSTEM_PROMPT_2,
}

# --- Load existing annotations ---
print(f"Loading existing annotations from {EXISTING_FILE}...")
existing_df = pd.read_csv(EXISTING_FILE)

# --- Load original comments ---
print(f"Loading original comments from {INPUT_FILE}...")
comments_df = pd.read_csv(INPUT_FILE)
comments = comments_df["text"].tolist()

# --- Find which rows need resubmitting per run ---
retry = {"run1": [], "run2": []}

for _, row in existing_df.iterrows():
    idx = int(row["index"])
    if row["label_run1"] in ["ERROR", "MISSING"]:
        retry["run1"].append(idx)
    if row["label_run2"] in ["ERROR", "MISSING"]:
        retry["run2"].append(idx)

print(f"Rows to retry — run1: {len(retry['run1'])} | run2: {len(retry['run2'])}")

# --- Client ---
client = anthropic.Anthropic()

# --- Submit retry batches ---
registry = []

for run_name, indices in retry.items():
    if not indices:
        print(f"No retries needed for {run_name}, skipping.")
        continue

    print(f"\n=== Submitting retry batches for {run_name} ({len(indices)} rows) ===")

    # Split into chunks
    chunks = [indices[i:i+BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]

    for chunk_idx, chunk in enumerate(chunks):
        requests = []
        for idx in chunk:
            if idx >= len(comments):
                print(f"  [SKIP] Index {idx} out of range")
                continue
            requests.append({
                "custom_id": f"{run_name}_{idx}",
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
                        {"role": "user", "content": comments[idx]}
                    ]
                }
            })

        batch = client.messages.batches.create(requests=requests)
        print(f"  Chunk {chunk_idx+1}/{len(chunks)} | batch_id={batch.id} | {len(requests)} requests")

        registry.append({
            "batch_id": batch.id,
            "run_name": run_name,
            "chunk": chunk_idx,
            "indices": chunk
        })

        time.sleep(2)

# --- Save registry ---
with open(REGISTRY_FILE, "w") as f:
    json.dump(registry, f, indent=2)

print(f"\nAll retry batches submitted. Registry saved to {REGISTRY_FILE}")
print("Now run retrieve_retry_SI.py when batches are complete (usually 1 hour).")
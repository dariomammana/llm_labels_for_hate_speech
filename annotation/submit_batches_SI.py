import anthropic
import json
import time
import pandas as pd

# --- Config ---
INPUT_FILE = "../Data/Unique_SI.csv"
REGISTRY_FILE = "batch_registry_SI.json"
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
print("Now run retrieve_results_SI.py when batches are complete (usually 1-2 hours).")

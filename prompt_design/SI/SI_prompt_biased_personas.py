import anthropic
import csv
import time
import pandas as pd

comments = pd.read_csv("Data/Sample150_SI.csv")["text"].tolist()

# --- Personas (Modified Text) ---
# Base definitions from Kralj Novak 2022
SCALE_DEFINITIONS = """3, NASILNO (VIOLENT), kar pomeni, da je usmerjeno v tarčo z nasilno vsebino. 
Lahko gre tudi za eksplicitno fizično nasilje ali implicitno nanašanje na zanikanje oziroma veličanje zgodovinskega nasilja.
2, ŽALJIVO (OFFENSIVE), usmerjeno v posameznike ali skupine, vključno z žaljivim posploševanjem, preziranjem, razčlovečenjem, posrednimi žaljivimi opazkami, vendar ni nasilno.
1, NEPRIMERNO (INAPPROPRIATE), če komentar nima tarče, vendar vsebuje izraze, ki so opsceni ali vulgarni, vendar niso nasilni ali žaljivi.
0, PRIMERNO (APPROPRIATE), če komentar ni usmerjen v nikogar in ne ustreza ničemur od zgoraj navedenega.

Odgovori SAMO z eno cifro: 0, 1, 2 ali 3. Nič drugega."""

# Persona 1: Simulates a highly tolerant internet user who ignores mild insults
SYSTEM_PROMPT_1 = """Ste visoko tolerantni klasifikator vsebine, ki simulira priložnostnega spletnega uporabnika z "debelo kožo".
Ravnajte previdno: ocene 1, 2 ali 3 dodelite le, če komentar jasno in nedvomno izpolnjuje merila. Če obstaja kakršna koli dvoumnost ali če se zdi, da gre za šalo, privzeto izberite nižjo oceno ali 0.""" + SCALE_DEFINITIONS

# Persona 2: Simulates a strict, highly sensitive content moderator
SYSTEM_PROMPT_2 = """Ste visoko občutljiv varnostni moderator, ki simulira strogo politiko platforme z ničelno toleranco.Pozorno iščite mikroagresije, skrita kodirana sporočila, prikrite pristranskosti in implicitno sovražnost. 
Če je komentar na meji ali bi ga kdorkoli lahko razumel kot žaljivega/neprimernega, ravnajte previdno in dodelite višjo možno oceno.""" + SCALE_DEFINITIONS

# Map runs to specific prompts
SYSTEM_PROMPTS = [SYSTEM_PROMPT_1, SYSTEM_PROMPT_2]

# --- SET TEMPERATURE HERE ---
# Force maximum variance between the two personas
TEMPERATURE = 1.0  

# --- Client ---
client = anthropic.Anthropic()

# --- Output file ---
OUTPUT_FILE = f"SI_prompt_biased_personas.csv"

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

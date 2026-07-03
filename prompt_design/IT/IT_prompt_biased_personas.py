import anthropic
import csv
import time
import pandas as pd

comments = pd.read_csv("Data/Sample150_IT.csv")["text"].tolist()

# --- Personas (Modified Text) ---
# Base definitions from Kralj Novak 2022
SCALE_DEFINITIONS = """3, VIOLENTO, ovvero che è rivolto a un bersaglio con contenuti violenti. Può trattarsi anche di violenza fisica esplicita o di un riferimento implicito alla negazione o glorificazione della violenza storica. 
2, OFFENSIVO, rivolto a individui o gruppi, comprese generalizzazioni offensive, disprezzo, disumanizzazione, osservazioni offensive indirette, ma non violente. 
1, INAPPROPRIATO, se il commento non ha un bersaglio ma contiene termini osceni o volgari, ma non violenti né offensivi. 
0, APPROPRIATO, se il commento non è rivolto a nessuno e non rientra in nessuno dei casi precedenti.

Rispondi solamente con una cifra singola: 0, 1, 2 o 3. Nient'altro."""

# Persona 1: Simulates a highly tolerant internet user who ignores mild insults
SYSTEM_PROMPT_1 = """Sei un classificatore per la moderazione dei contenuti altamente tollerante che simula un utente di internet informale e dalla pelle dura.
Sbaglia per eccesso di cautela: assegna i punteggi 1, 2 o 3 solo se il commento soddisfa i criteri in modo chiaro e innegabile. In caso di ambiguità o se sembra uno scherzo, passa per impostazione predefinita a un punteggio inferiore o a 0.""" + SCALE_DEFINITIONS
# Persona 2: Simulates a strict, highly sensitive content moderator
SYSTEM_PROMPT_2 = """Sei un moderatore della sicurezza altamente sensibile che simula una politica di piattaforma rigida e a tolleranza zero.
Cerca attentamente microaggressioni, messaggi in codice, pregiudizi nascosti e ostilità implicita. Se un commento è al limite o potrebbe essere interpretato come offensivo/inappropriato da chiunque, sbaglia per eccesso di cautela e assegna il punteggio applicabile più alto.""" + SCALE_DEFINITIONS
# Map runs to specific prompts
SYSTEM_PROMPTS = [SYSTEM_PROMPT_1, SYSTEM_PROMPT_2]

# --- SET TEMPERATURE HERE ---
# Force maximum variance between the two personas
TEMPERATURE = 1.0  

# --- Client ---
client = anthropic.Anthropic()

# --- Output file ---
OUTPUT_FILE = f"IT_prompt_biased_personas.csv"

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

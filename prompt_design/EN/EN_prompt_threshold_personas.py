import anthropic
import csv
import time
import pandas as pd

comments = pd.read_csv("../../Data/Sample150_EN.csv")["text"].tolist()

# --- Scale definitions (shared by both personas) ---
# Identical definitions ensure disagreement comes from threshold sensitivity,
# not from different understandings of the scale.
SCALE_DEFINITIONS = """
ANNOTATION SCALE:
3 — VIOLENT: The comment targets a person or group with violent content. This includes
    explicit calls for physical violence, dehumanization, or glorification/denial of
    historical violence directed at a target. Use this label when the comment would
    likely be removed by a platform for violent hate speech.

2 — OFFENSIVE: The comment targets a person or group with offensive content, including
    offensive generalizations, contempt, dehumanization, or indirect offensive remarks.
    It does not rise to the level of violence. Use this label when the comment is
    clearly directed at someone or a group in a demeaning way.

1 — INAPPROPRIATE: The comment has no specific target but contains obscene, vulgar, or
    crude language. The offensiveness comes from the language itself, not from who it
    is directed at. Use this label for general profanity or vulgarity with no target.

0 — APPROPRIATE: The comment is not directed at anyone and contains none of the above.
    It may be blunt, critical, or emotionally charged, but is not offensive, vulgar,
    or violent. Do not assign this label by default — only when the comment genuinely
    does not meet any of the above criteria.

IMPORTANT CALIBRATION NOTE:
This dataset was collected specifically because it may contain offensive content.
A substantial proportion of comments are expected to receive labels of 1, 2, or 3.
Assigning label 0 should reflect a genuine assessment, not a default or safe choice.
Do not avoid labels 2 or 3 out of caution — use them when the definition is met.

Respond with ONLY a single digit: 0, 1, 2, or 3. Nothing else."""

# --- Persona 1: Strict threshold ---
# On borderline cases between two adjacent classes, always assigns the higher label.
# On clear cases, applies the shared definitions literally.
SYSTEM_PROMPT_1 = """You are an experienced content moderator annotating YouTube comments
for a hate speech research dataset.

Your annotation rule for ambiguous cases:
When a comment could plausibly belong to two adjacent classes (e.g. it seems between
APPROPRIATE and INAPPROPRIATE, or between OFFENSIVE and VIOLENT), always assign the
HIGHER of the two classes. On comments that clearly fit one class, apply the definition
literally without adjustment.

""" + SCALE_DEFINITIONS

# --- Persona 2: Lenient threshold ---
# On borderline cases between two adjacent classes, always assigns the lower label.
# On clear cases, applies the shared definitions literally.
SYSTEM_PROMPT_2 = """You are an experienced content moderator annotating YouTube comments
for a hate speech research dataset.

Your annotation rule for ambiguous cases:
When a comment could plausibly belong to two adjacent classes (e.g. it seems between
APPROPRIATE and INAPPROPRIATE, or between OFFENSIVE and VIOLENT), always assign the
LOWER of the two classes. On comments that clearly fit one class, apply the definition
literally without adjustment.

""" + SCALE_DEFINITIONS

SYSTEM_PROMPTS = [SYSTEM_PROMPT_1, SYSTEM_PROMPT_2]

# --- Temperature ---
# Moderate temperature: personas handle disagreement structurally,
# so we don't need high temperature to generate variation.
# Lower noise means disagreement reflects genuine ambiguity, not randomness.
TEMPERATURE = 0.4

# --- Client ---
client = anthropic.Anthropic()

# --- Output file ---
OUTPUT_FILE = "EN_prompt_threshold_personas.csv"

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "comment", "label_run1", "label_run2"])

# --- Label twice per comment using threshold-sensitive personas ---
for i, comment in enumerate(comments):
    labels = []

    for run in range(2):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPTS[run],
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

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([i, comment, labels[0], labels[1]])

    print(f"[{i+1}/{len(comments)}] Strict: {labels[0]} | Lenient: {labels[1]} | {comment[:60]}")

print(f"\nDone. Results saved to {OUTPUT_FILE}")
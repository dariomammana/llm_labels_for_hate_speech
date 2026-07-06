import anthropic
import json
import time
import pandas as pd

# --- Config ---
REGISTRY_FILE = "batch_registry_SI_retry.json"
EXISTING_FILE = "../Data/SI_annotations.csv"
OUTPUT_FILE = "../Data/SI_annotations.csv"
UNIQUE_FILE = "../Data/Unique_SI.csv"
POLL_INTERVAL = 60

# --- Client ---
client = anthropic.Anthropic()

# --- Load unique texts ---
print(f"Loading unique texts from {UNIQUE_FILE}...")
unique_df = pd.read_csv(UNIQUE_FILE)
comments = unique_df["text"].tolist()
print(f"Found {len(comments)} unique texts.")

# --- Load registry ---
print(f"Loading retry registry from {REGISTRY_FILE}...")
with open(REGISTRY_FILE, "r") as f:
    registry = json.load(f)
print(f"Found {len(registry)} retry batches.\n")

# --- Poll until all complete ---
print("Checking batch statuses...")
while True:
    complete = True
    for entry in registry:
        batch = client.messages.batches.retrieve(entry["batch_id"])
        counts = batch.request_counts
        print(f"  {entry['run_name']} chunk {entry['chunk']} | "
              f"status={batch.processing_status} | "
              f"succeeded={counts.succeeded} | "
              f"errored={counts.errored}")
        if batch.processing_status != "ended":
            complete = False

    if complete:
        print("\nAll retry batches complete. Retrieving results...\n")
        break
    else:
        print(f"  Not done yet — checking again in {POLL_INTERVAL}s...\n")
        time.sleep(POLL_INTERVAL)

# --- Retrieve retry results ---
new_results = {}

for entry in registry:
    batch_id = entry["batch_id"]
    run_name = entry["run_name"]
    print(f"Retrieving {run_name} chunk {entry['chunk']} | batch_id={batch_id}")

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        idx = int(custom_id.split("_")[1])

        if result.result.type == "succeeded":
            label = result.result.message.content[0].text.strip()
            if label not in ["0", "1", "2", "3"]:
                print(f"  [WARNING] Unexpected label for {custom_id}: '{label}' — marking ERROR")
                label = "ERROR"
        else:
            print(f"  [FAILED] {custom_id}: {result.result.error}")
            label = "ERROR"

        if idx not in new_results:
            new_results[idx] = {}
        new_results[idx][run_name] = label

print(f"\nRetrieved {len(new_results)} retry results.")

# --- Load existing annotations ---
print(f"Loading existing annotations from {EXISTING_FILE}...")
existing_df = pd.read_csv(EXISTING_FILE)

# --- Add text column if not already present ---
if "text" not in existing_df.columns:
    existing_df["text"] = [
        comments[idx] if idx < len(comments) else ""
        for idx in existing_df["index"]
    ]

# --- Merge retry results into existing ---
updated = 0
for idx, runs in new_results.items():
    for run_name, label in runs.items():
        col = "label_run1" if run_name == "run1" else "label_run2"
        existing_df.loc[existing_df["index"] == idx, col] = label
        updated += 1

print(f"Updated {updated} cells in existing annotations.")

# --- Reorder columns and save ---
existing_df = existing_df[["index", "text", "label_run1", "label_run2"]]
existing_df.to_csv(OUTPUT_FILE, index=False)

missing = existing_df[(existing_df["label_run1"] == "MISSING") | (existing_df["label_run2"] == "MISSING")]
errors  = existing_df[(existing_df["label_run1"] == "ERROR")   | (existing_df["label_run2"] == "ERROR")]

print(f"\nDone. {len(existing_df)} rows written to {OUTPUT_FILE}")
print(f"Still missing: {len(missing)} | Still errors: {len(errors)}")
if len(errors) > 0:
    print("Remaining errors are likely problematic comments (URLs, empty strings) — safe to drop.")
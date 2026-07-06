import anthropic
import json
import time
import pandas as pd

# --- Config ---
REGISTRY_FILE = "batch_registry_EN.json"
OUTPUT_FILE = "../Data/EN_annotations.csv"
UNIQUE_FILE = "../Data/Unique_EN.csv"
POLL_INTERVAL = 60  # seconds between status checks

# --- Client ---
client = anthropic.Anthropic()

# --- Load registry ---
print(f"Loading registry from {REGISTRY_FILE}...")
with open(REGISTRY_FILE, "r") as f:
    registry = json.load(f)
print(f"Found {len(registry)} batches to retrieve.\n")

print(f"Loading unique texts from {UNIQUE_FILE}...")
unique_df = pd.read_csv(UNIQUE_FILE)
comments = unique_df["text"].tolist()
print(f"Found {len(comments)} unique texts.")

# --- Poll until all batches are complete ---
def all_complete(registry):
    for entry in registry:
        batch = client.messages.batches.retrieve(entry["batch_id"])
        if batch.processing_status != "ended":
            return False, batch
    return True, None

print("Checking batch statuses...")
while True:
    complete = True
    for entry in registry:
        batch = client.messages.batches.retrieve(entry["batch_id"])
        counts = batch.request_counts
        status = batch.processing_status
        print(f"  {entry['run_name']} chunk starting at {entry['start_index']:>6} | "
              f"status={status} | "
              f"processing={counts.processing} "
              f"succeeded={counts.succeeded} "
              f"errored={counts.errored}")
        if status != "ended":
            complete = False

    if complete:
        print("\nAll batches complete. Retrieving results...\n")
        break
    else:
        print(f"  Not all done yet — checking again in {POLL_INTERVAL}s...\n")
        time.sleep(POLL_INTERVAL)

# --- Retrieve results ---
all_results = {}  # {index: {"run1": label, "run2": label}}

for entry in registry:
    batch_id = entry["batch_id"]
    run_name = entry["run_name"]
    print(f"Retrieving {run_name} | start_index={entry['start_index']} | batch_id={batch_id}")

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id          # e.g. "run1_42"
        idx = int(custom_id.split("_")[1])

        if result.result.type == "succeeded":
            label = result.result.message.content[0].text.strip()
            if label not in ["0", "1", "2", "3"]:
                print(f"  [WARNING] Unexpected label for {custom_id}: '{label}' — marking ERROR")
                label = "ERROR"
        else:
            print(f"  [ERROR] {custom_id} failed: {result.result.error}")
            label = "ERROR"

        if idx not in all_results:
            all_results[idx] = {}
        all_results[idx][run_name] = label

# --- Determine total expected comments from registry ---
total_comments = max(
    entry["start_index"] + entry["size"]
    for entry in registry
)

# --- Write output CSV ---
print(f"\nWriting results to {OUTPUT_FILE}...")
rows = []
for idx in range(total_comments):
    run1 = all_results.get(idx, {}).get("run1", "MISSING")
    run2 = all_results.get(idx, {}).get("run2", "MISSING")
    text = comments[idx] if idx < len(comments) else ""
    rows.append({
        "index": idx,
        "text": text,
        "label_run1": run1,
        "label_run2": run2,
    })

output_df = pd.DataFrame(rows)
output_df.to_csv(OUTPUT_FILE, index=False)

# --- Summary ---
missing = output_df[(output_df["label_run1"] == "MISSING") | (output_df["label_run2"] == "MISSING")]
errors  = output_df[(output_df["label_run1"] == "ERROR")   | (output_df["label_run2"] == "ERROR")]

print(f"Done. {len(output_df)} rows written to {OUTPUT_FILE}")
print(f"Missing: {len(missing)} | Errors: {len(errors)}")
if len(missing) > 0 or len(errors) > 0:
    print("Some rows have issues — check batch_registry_EN.json and resubmit if needed.")
else:
    print("All labels retrieved successfully.")

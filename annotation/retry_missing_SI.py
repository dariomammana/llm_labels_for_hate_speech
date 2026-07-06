import anthropic
import json
import pandas as pd

# --- Config ---
REGISTRY_FILE = "batch_registry_SI.json"
EXISTING_FILE = "../Data/SI_annotations.csv"
OUTPUT_FILE = "../Data/SI_annotations.csv"  # overwrites with fixed version

# --- Load existing results ---
print(f"Loading existing results from {EXISTING_FILE}...")
existing_df = pd.read_csv(EXISTING_FILE)

# Find indices with errors in either run
error_mask = (existing_df["label_run1"] == "ERROR") | (existing_df["label_run2"] == "ERROR")
missing_mask = (existing_df["label_run1"] == "MISSING") | (existing_df["label_run2"] == "MISSING")
problem_indices = set(existing_df[error_mask | missing_mask]["index"].tolist())
print(f"Rows needing retry: {len(problem_indices)}")

# Build lookup from existing good results
results = {}
for _, row in existing_df.iterrows():
    results[row["index"]] = {
        "run1": row["label_run1"],
        "run2": row["label_run2"]
    }

# --- Client ---
client = anthropic.Anthropic()

# --- Load registry ---
print(f"\nLoading registry from {REGISTRY_FILE}...")
with open(REGISTRY_FILE, "r") as f:
    registry = json.load(f)

# --- Re-retrieve only problem rows from each batch ---
for entry in registry:
    batch_id = entry["batch_id"]
    run_name = entry["run_name"]
    start = entry["start_index"]
    size = entry["size"]

    # Check if this batch overlaps with any problem indices
    batch_indices = set(range(start, start + size))
    relevant = batch_indices & problem_indices
    if not relevant:
        print(f"Skipping {run_name} start={start} — no problem rows in this chunk")
        continue

    print(f"\nRe-retrieving {run_name} | start={start} | {len(relevant)} problem rows to fix...")

    try:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            idx = int(custom_id.split("_")[1])

            if idx not in problem_indices:
                continue  # skip rows that are already fine

            if result.result.type == "succeeded":
                label = result.result.message.content[0].text.strip()
                if label not in ["0", "1", "2", "3"]:
                    print(f"  [WARNING] Unexpected label for {custom_id}: '{label}' — marking ERROR")
                    label = "ERROR"
            else:
                print(f"  [FAILED] {custom_id}: {result.result.error}")
                label = "ERROR"

            if idx not in results:
                results[idx] = {}
            results[idx][run_name] = label

    except Exception as e:
        print(f"  [ERROR] Could not retrieve batch {batch_id}: {e}")

# --- Write fixed output ---
print(f"\nWriting fixed results to {OUTPUT_FILE}...")
total = max(results.keys()) + 1
rows = []
for idx in range(total):
    run1 = results.get(idx, {}).get("run1", "MISSING")
    run2 = results.get(idx, {}).get("run2", "MISSING")
    rows.append({"index": idx, "label_run1": run1, "label_run2": run2})

output_df = pd.DataFrame(rows)
output_df.to_csv(OUTPUT_FILE, index=False)

missing = output_df[(output_df["label_run1"] == "MISSING") | (output_df["label_run2"] == "MISSING")]
errors  = output_df[(output_df["label_run1"] == "ERROR")   | (output_df["label_run2"] == "ERROR")]

print(f"Done. {len(output_df)} rows written to {OUTPUT_FILE}")
print(f"Still missing: {len(missing)} | Still errors: {len(errors)}")
if len(missing) > 0:
    print("Missing indices:", missing["index"].tolist()[:20], "...")
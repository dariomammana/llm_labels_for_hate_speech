import pandas as pd
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Data" / "Test_data"

FILES = {
    "IMSyPP_EN_YouTube_comments_evaluation_no_context.csv": {
        "text_col": "Text",
        "type_col": "Type",
        "mapping": {
            "0. appropriate": 0,
            "1. inappropriate": 1,
            "2. offensive": 2,
            "3. violent": 3,
        },
    },
    "IMSyPP_IT_YouTube_comments_evaluation.csv": {
        "text_col": "Testo",
        "type_col": "Tipo",
        "mapping": {
            "0. appropriato": 0,
            "1. inappropriato": 1,
            "2. offensivo": 2,
            "3. violento": 3,
        },
    },
    "IMSyPP_SI_anotacije_round2.csv": {
        "text_col": "besedilo",
        "type_col": "vrsta",
        "mapping": {
            "0 ni sporni govor": 0,
            "1 nespodobni govor": 1,
            "2 žalitev": 2,
            "3 nasilje": 3,
        },
    },
}

for filename, config in FILES.items():
    input_file = INPUT_DIR / filename

    print(f"Processing {filename}...")

    df = pd.read_csv(input_file)

    # Create output dataframe
    output_df = pd.DataFrame({
        "index": range(len(df)),
        "text": df[config["text_col"]],
        "label": (
            df[config["type_col"]]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({k.lower(): v for k, v in config["mapping"].items()})
        ),
    })

    # Check for unmapped labels
    unmapped = output_df["label"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} labels could not be mapped in {filename}")

    output_file = input_file.with_name(
        input_file.stem + "_mapped.csv"
    )

    output_df.to_csv(output_file, index=False)

    print(f"  Saved: {output_file}")

print("Done.")
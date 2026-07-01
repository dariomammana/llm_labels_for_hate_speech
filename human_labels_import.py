from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


SOURCE_FILES = {
	"EN": {
		"input": DATA_DIR / "IMSyPP_EN_YouTube_comments_train.csv",
		"output": DATA_DIR / "Unique_EN.csv",
		"text_column": "Text",
		"drop_columns": ["Type", "Target", "Annotator"],
	},
	"IT": {
		"input": DATA_DIR / "IMSyPP_IT_YouTube_comments_train.csv",
		"output": DATA_DIR / "Unique_IT.csv",
		"text_column": "Testo",
		"drop_columns": ["Tipo", "Target"],
	},
	"SI": {
		"input": DATA_DIR / "IMSyPP_SI_anotacije_round1(in).csv",
		"output": DATA_DIR / "Unique_SI.csv",
		"text_column": "besedilo",
		"drop_columns": ["vrsta", "uporabnik", "tarča", "jezik/kontekst", "Annotator"],
	},
}


def build_unique_csv(input_path: Path, output_path: Path, text_column: str, drop_columns: list[str]) -> pd.DataFrame:
	"""Build a text-only CSV containing all unique comments from one source file.

	The uniqueness criterion is the text column, and the output intentionally
	keeps only that text column.
	"""
	dataframe = pd.read_csv(input_path, usecols=[text_column], low_memory=False)
	dataframe = dataframe.dropna(subset=[text_column]).copy()
	dataframe = dataframe.drop_duplicates(subset=[text_column], keep="first")
	dataframe = dataframe[[text_column]].rename(columns={text_column: "text"})
	dataframe.to_csv(output_path, index=False)
	return dataframe


def _parse_type_value(value: object) -> int:
	"""Convert labels like '0. appropriate' or '2 violento' into integers.

	If your source labels change later, adjust this parser first.
	"""
	match = re.match(r"^\s*([0-3])", str(value))
	if not match:
		return pd.NA
	return int(match.group(1))


def build_sampled_csv(
	input_path: Path,
	output_path: Path,
	text_column: str,
	type_column: str,
	total_rows: int = 150,
	minimum_per_type: int = 15,
	random_seed: int = 42,
) -> pd.DataFrame:
	"""Write a 150-row sample of unique comments with two type labels per row.

	Each row includes both available annotations (type_1 and type_2) for inter-annotator agreement.

	Tuning knobs:
	- total_rows: total rows in the output file.
	- minimum_per_type: minimum rows for each type category 0..3.
	- random_seed: seed used to make the remaining random fill reproducible.
	"""
	dataframe = pd.read_csv(input_path, usecols=[text_column, type_column], low_memory=False)
	if text_column not in dataframe.columns:
		raise KeyError(f"Missing text column {text_column!r} in {input_path.name}")
	if type_column not in dataframe.columns:
		raise KeyError(f"Missing type column {type_column!r} in {input_path.name}")

	working = dataframe.dropna(subset=[text_column, type_column]).copy()
	working["_row_order"] = range(len(working))
	working["_type_int"] = working[type_column].map(_parse_type_value)
	working = working.dropna(subset=["_type_int"]).copy()
	working["_type_int"] = working["_type_int"].astype(int)

	# Group by text to get all annotations per text
	grouped = working.groupby(text_column)["_type_int"].apply(list).reset_index()
	grouped.columns = [text_column, "_types"]
	
	# Filter to texts that have at least 2 annotations
	grouped = grouped[grouped["_types"].apply(len) >= 2].copy()
	grouped["_type_int"] = grouped["_types"].apply(lambda x: x[0])  # Use first type for stratification
	
	selected_parts = []
	selected_texts = set()

	for type_value in range(4):
		type_rows = grouped[grouped["_type_int"] == type_value]
		if len(type_rows) < minimum_per_type:
			raise ValueError(
				f"{input_path.name} has only {len(type_rows)} texts with at least 2 annotations for type {type_value}; "
				f"need at least {minimum_per_type}."
			)
		sampled_type_rows = type_rows.sample(n=minimum_per_type, random_state=random_seed + type_value)
		selected_parts.append(sampled_type_rows)
		selected_texts.update(sampled_type_rows[text_column].tolist())

	selected_df = pd.concat(selected_parts, ignore_index=True)
	remaining_pool = grouped[~grouped[text_column].isin(selected_texts)]
	remaining_needed = total_rows - len(selected_df)
	if remaining_needed < 0:
		raise ValueError(
			f"Minimum per type exceeds target size for {input_path.name}: "
			f"{len(selected_df)} rows selected before random fill."
		)
	if len(remaining_pool) < remaining_needed:
		raise ValueError(
			f"{input_path.name} has only {len(grouped)} texts with 2+ annotations; "
			f"need {total_rows}."
		)

	remaining_df = remaining_pool.sample(n=remaining_needed, random_state=random_seed)
	final_df = pd.concat([selected_df, remaining_df], ignore_index=True)
	final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
	
	# Extract type_1 and type_2 from the _types list
	final_df["type_1"] = final_df["_types"].apply(lambda x: x[0])
	final_df["type_2"] = final_df["_types"].apply(lambda x: x[1] if len(x) > 1 else x[0])
	final_df = final_df[[text_column, "type_1", "type_2"]].rename(columns={text_column: "text"})
	final_df.to_csv(output_path, index=False)
	return final_df


en_df = build_unique_csv(
	SOURCE_FILES["EN"]["input"],
	SOURCE_FILES["EN"]["output"],
	SOURCE_FILES["EN"]["text_column"],
	SOURCE_FILES["EN"]["drop_columns"],
)
it_df = build_unique_csv(
	SOURCE_FILES["IT"]["input"],
	SOURCE_FILES["IT"]["output"],
	SOURCE_FILES["IT"]["text_column"],
	SOURCE_FILES["IT"]["drop_columns"],
)
si_df = build_unique_csv(
	SOURCE_FILES["SI"]["input"],
	SOURCE_FILES["SI"]["output"],
	SOURCE_FILES["SI"]["text_column"],
	SOURCE_FILES["SI"]["drop_columns"],
)


SAMPLED_FILES = {
	"EN": DATA_DIR / "Sample150_EN.csv",
	"IT": DATA_DIR / "Sample150_IT.csv",
	"SI": DATA_DIR / "Sample150_SI.csv",
}


def build_sampled_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Create the 150-row sample files.

	If you want to change the sample size or duplicate rule, edit build_sampled_csv.
	"""
	en_sample_df = build_sampled_csv(
		SOURCE_FILES["EN"]["input"],
		SAMPLED_FILES["EN"],
		SOURCE_FILES["EN"]["text_column"],
		"Type",
	)
	it_sample_df = build_sampled_csv(
		SOURCE_FILES["IT"]["input"],
		SAMPLED_FILES["IT"],
		SOURCE_FILES["IT"]["text_column"],
		"Tipo",
	)
	si_sample_df = build_sampled_csv(
		SOURCE_FILES["SI"]["input"],
		SAMPLED_FILES["SI"],
		SOURCE_FILES["SI"]["text_column"],
		"vrsta",
	)
	return en_sample_df, it_sample_df, si_sample_df


if __name__ == "__main__":
	build_sampled_outputs()
	print(f"Wrote {SOURCE_FILES['EN']['output'].name}")
	print(f"Wrote {SOURCE_FILES['IT']['output'].name}")
	print(f"Wrote {SOURCE_FILES['SI']['output'].name}")
	print(f"Wrote {SAMPLED_FILES['EN'].name}")
	print(f"Wrote {SAMPLED_FILES['IT'].name}")
	print(f"Wrote {SAMPLED_FILES['SI'].name}")

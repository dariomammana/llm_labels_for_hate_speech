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
	dataframe = pd.read_csv(input_path, low_memory=False)
	dataframe = dataframe.drop(columns=drop_columns, errors="ignore")
	dataframe = dataframe.drop_duplicates(subset=[text_column], keep="first")
	dataframe = dataframe[[text_column]]
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
	repeated_texts: int = 5,
	minimum_per_type: int = 5,
) -> pd.DataFrame:
	"""Write a 150-row sample with controlled duplicates and integer types.

	Tuning knobs:
	- total_rows: total rows in the output file.
	- repeated_texts: number of comments that are intentionally repeated twice.
	- minimum_per_type: minimum rows for each type category 0..3.
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

	duplicate_candidates = (
		working.groupby(text_column)["_type_int"]
		.nunique()
		.sort_index()
	)
	repeated_text_values = []
	for text_value in working.sort_values("_row_order")[text_column].drop_duplicates():
		if duplicate_candidates.get(text_value, 0) > 1:
			repeated_text_values.append(text_value)
		if len(repeated_text_values) == repeated_texts:
			break

	if len(repeated_text_values) < repeated_texts:
		raise ValueError(
			f"Only found {len(repeated_text_values)} comments with multiple types in {input_path.name}; "
			f"need {repeated_texts}."
		)

	repeated_rows = []
	for text_value in repeated_text_values:
		text_rows = working[working[text_column] == text_value].sort_values("_row_order")
		seen_types = set()
		for _, row in text_rows.iterrows():
			if row["_type_int"] in seen_types:
				continue
			repeated_rows.append(row)
			seen_types.add(row["_type_int"])
			if len(seen_types) == 2:
				break
		if len(seen_types) < 2:
			raise ValueError(
				f"Comment {text_value!r} does not have two distinct type values in {input_path.name}."
			)

	repeated_df = pd.DataFrame(repeated_rows)
	selected_texts = set(repeated_text_values)
	selected_unique_rows = []
	selected_unique_count = total_rows - len(repeated_df)

	repeated_type_counts = repeated_df["_type_int"].value_counts().to_dict()
	base_unique = working.drop_duplicates(subset=[text_column], keep="first")
	base_unique = base_unique[~base_unique[text_column].isin(selected_texts)].sort_values("_row_order")

	# First satisfy the minimum type quota using the first available unique text for each type.
	for type_value in range(4):
		needed = max(0, minimum_per_type - repeated_type_counts.get(type_value, 0))
		if needed == 0:
			continue
		matching_rows = base_unique[base_unique["_type_int"] == type_value]
		for _, row in matching_rows.iterrows():
			if row[text_column] in selected_texts:
				continue
			selected_unique_rows.append(row)
			selected_texts.add(row[text_column])
			needed -= 1
			if needed == 0:
				break
		if needed > 0:
			raise ValueError(
				f"Not enough rows of type {type_value} in {input_path.name} to reach {minimum_per_type}."
			)

	# Then fill the remaining slots in original order, keeping one row per text.
	for _, row in base_unique.iterrows():
		if len(selected_unique_rows) >= selected_unique_count:
			break
		if row[text_column] in selected_texts:
			continue
		selected_unique_rows.append(row)
		selected_texts.add(row[text_column])

	if len(selected_unique_rows) < selected_unique_count:
		raise ValueError(
			f"Only built {len(selected_unique_rows)} unique rows for {input_path.name}; "
			f"need {selected_unique_count}."
		)

	selected_unique_df = pd.DataFrame(selected_unique_rows)
	final_df = pd.concat([repeated_df, selected_unique_df], ignore_index=True)
	final_df = final_df.sort_values("_row_order").reset_index(drop=True)
	final_df = final_df[[text_column, "_type_int"]].rename(columns={"_type_int": "Type"})
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

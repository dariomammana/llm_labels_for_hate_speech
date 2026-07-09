from pathlib import Path
import re

import pandas as pd

import krippendorff
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


SOURCE_FILES = {
	"EN": {
		"input": DATA_DIR / "IMSyPP_EN_YouTube_comments_train.csv",
		"text_column": "Text",
		"type_column": "Type",
	},
	"IT": {
		"input": DATA_DIR / "IMSyPP_IT_YouTube_comments_train.csv",
		"text_column": "Testo",
		"type_column": "Tipo",
	},
	"SI": {
		"input": DATA_DIR / "IMSyPP_SI_anotacije_round1(in).csv",
		"text_column": "besedilo",
		"type_column": "vrsta",
	},
}


TYPE_NAMES = {
	0: "appropriate",
	1: "inappropriate",
	2: "offensive",
	3: "violent",
}


def parse_type_value(value: object) -> int:
	"""Convert label strings like '0. appropriate' into an integer type id.

	If the source format changes later, adjust this parser first.
	"""
	match = re.match(r"^\s*([0-3])", str(value))
	if not match:
		return pd.NA
	return int(match.group(1))


def load_type_counts(input_path: Path, type_column: str) -> tuple[int, pd.Series, int]:
	"""Load one source file and return total valid comments, counts, and skipped rows."""
	dataframe = pd.read_csv(input_path, usecols=[type_column], low_memory=False)
	type_series = dataframe[type_column].dropna().map(parse_type_value)
	valid_series = type_series.dropna().astype(int)
	skipped_rows = len(type_series) - len(valid_series)
	counts = valid_series.value_counts().reindex(range(4), fill_value=0).sort_index()
	return len(valid_series), counts, skipped_rows


def load_repeated_annotations(input_path: Path, text_column: str, type_column: str) -> list[list[int]]:
	"""Return one annotation list per unique comment text.

	Each list contains the valid type ids observed for that text. Singletons are
	kept out of the result because they do not contribute to agreement.
	"""
	dataframe = pd.read_csv(input_path, usecols=[text_column, type_column], low_memory=False)
	dataframe = dataframe.dropna(subset=[text_column, type_column]).copy()
	dataframe["_type_int"] = dataframe[type_column].map(parse_type_value)
	dataframe = dataframe.dropna(subset=["_type_int"]).copy()
	dataframe["_type_int"] = dataframe["_type_int"].astype(int)

	annotations = []
	for _, group in dataframe.groupby(text_column, sort=False):
		values = group["_type_int"].tolist()
		if len(values) >= 2:
			annotations.append(values)
	return annotations


def krippendorff_ordinal_alpha(annotations: list[list[int]]) -> float:
    if not annotations:
        return float("nan")

    max_raters = max(len(unit) for unit in annotations)

    data = np.full((max_raters, len(annotations)), np.nan)

    for col, unit in enumerate(annotations):
        for row, label in enumerate(unit):
            data[row, col] = label

    return float(
        krippendorff.alpha(
            reliability_data=data,
            level_of_measurement="ordinal",
        )
    )

def print_distribution(name: str, input_path: Path, type_column: str) -> None:
	"""Print the total number of comments and the share per type for one file."""
	total_comments, counts, skipped_rows = load_type_counts(input_path, type_column)
	print(f"{name} ({input_path.name})")
	print(f"  valid comments: {total_comments}")
	if skipped_rows:
		print(f"  skipped rows without a valid type label: {skipped_rows}")
	for type_id in range(4):
		count = int(counts.loc[type_id])
		share = (count / total_comments * 100) if total_comments else 0.0
		print(f"  {type_id} {TYPE_NAMES[type_id]}: {count} ({share:.2f}%)")
	print()


def print_ordinal_alpha(name: str, input_path: Path, text_column: str, type_column: str) -> None:
	"""Print Krippendorff's ordinal alpha for repeated unique comments."""
	annotations = load_repeated_annotations(input_path, text_column, type_column)
	alpha = krippendorff_ordinal_alpha(annotations)
	print(f"{name} ordinal alpha ({input_path.name})")
	print(f"  repeated unique comments used: {len(annotations)}")
	print(f"  krippendorff ordinal alpha: {alpha:.4f}")
	print()


def main() -> None:
	"""Print the type distribution for the original EN, IT, and SI CSV files.

	To change the label mapping, edit parse_type_value or TYPE_NAMES.
	To change the alpha behavior, edit load_repeated_annotations or
	krippendorff_ordinal_alpha.
	"""
	for name, config in SOURCE_FILES.items():
		print_distribution(name, config["input"], config["type_column"])
		print_ordinal_alpha(name, config["input"], config["text_column"], config["type_column"])


if __name__ == "__main__":
	main()
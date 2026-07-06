from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


SOURCE_FILES = {
	"EN": {
		"source": DATA_DIR / "IMSyPP_EN_YouTube_comments_train.csv",
		"unique": DATA_DIR / "Unique_EN.csv",
		"output": DATA_DIR / "EN_human_annotations.csv",
		"text_column": "Text",
		"label_column": "Type",
	},
	"IT": {
		"source": DATA_DIR / "IMSyPP_IT_YouTube_comments_train.csv",
		"unique": DATA_DIR / "Unique_IT.csv",
		"output": DATA_DIR / "IT_human_annotations.csv",
		"text_column": "Testo",
		"label_column": "Tipo",
	},
	"SI": {
		"source": DATA_DIR / "IMSyPP_SI_anotacije_round1(in).csv",
		"unique": DATA_DIR / "Unique_SI.csv",
		"output": DATA_DIR / "SI_human_annotations.csv",
		"text_column": "besedilo",
		"label_column": "vrsta",
	},
}


def parse_label(value: object) -> int | None:
	text_value = str(value).strip()
	if not text_value:
		return None
	parts = text_value.split()
	if not parts:
		return None
	# The IMSyPP labels are formatted like "0. appropriate" or "2 offensivo";
	# keep only the leading numeric class id.
	first_part = parts[0].rstrip(".,;:")
	if first_part.isdigit():
		label_value = int(first_part)
		if 0 <= label_value <= 3:
			return label_value
	return None


def add_index_column(csv_path: Path) -> bool:
	dataframe = pd.read_csv(csv_path)
	if "index" in dataframe.columns:
		return False

	dataframe.insert(0, "index", range(1, len(dataframe) + 1))
	dataframe.to_csv(csv_path, index=False)
	return True


def build_human_annotations_file(language: str, config: dict[str, Path | str]) -> Path:
	unique_df = pd.read_csv(config["unique"])
	if not {"index", "text"}.issubset(unique_df.columns):
		raise ValueError(f"{config['unique'].name} must contain 'index' and 'text' columns")

	source_df = pd.read_csv(config["source"], usecols=[config["text_column"], config["label_column"]], low_memory=False)
	source_df = source_df.dropna(subset=[config["text_column"], config["label_column"]]).copy()
	source_df["_label_int"] = source_df[config["label_column"]].map(parse_label)
	source_df = source_df.dropna(subset=["_label_int"]).copy()
	source_df["_label_int"] = source_df["_label_int"].astype(int)

	grouped = source_df.groupby(config["text_column"])["_label_int"].apply(list).to_dict()

	output_df = unique_df.copy()
	output_df["index"] = output_df["index"].astype(int) - 1
	output_df["label_run1"] = output_df["text"].map(lambda text: grouped.get(text, [pd.NA]))
	output_df["label_run1"] = output_df["label_run1"].map(lambda labels: labels[0] if isinstance(labels, list) and len(labels) > 0 else pd.NA)
	output_df["label_run2"] = output_df["text"].map(lambda text: grouped.get(text, [pd.NA]))
	output_df["label_run2"] = output_df["label_run2"].map(
		lambda labels: labels[1] if isinstance(labels, list) and len(labels) > 1 else (labels[0] if isinstance(labels, list) and len(labels) > 0 else pd.NA)
	)

	output_df = output_df[["index", "text", "label_run1", "label_run2"]]
	output_df.to_csv(config["output"], index=False)
	return config["output"]


def main() -> None:
	unique_files = sorted(DATA_DIR.glob("Unique_*.csv"))
	if not unique_files:
		print("No Unique_*.csv files found.")
		return

	for csv_path in unique_files:
		updated = add_index_column(csv_path)
		status = "updated" if updated else "skipped (already has index column)"
		print(f"{csv_path.name}: {status}")

	for language, config in SOURCE_FILES.items():
		output_path = build_human_annotations_file(language, config)
		print(f"{output_path.name}: written")


if __name__ == "__main__":
	main()

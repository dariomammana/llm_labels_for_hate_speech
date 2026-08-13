from __future__ import annotations

from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


SOURCE_GROUPS = {
	"human": [
		DATA_DIR / "EN_human_annotations.csv",
		DATA_DIR / "IT_human_annotations.csv",
		DATA_DIR / "SI_human_annotations.csv",
	],
	"LLM": [
		DATA_DIR / "EN_LLM_annotations.csv",
		DATA_DIR / "IT_LLM_annotations.csv",
		DATA_DIR / "SI_LLM_annotations.csv",
	],
}

PAIRWISE_LANGUAGE_PAIRS = {
	"EN": (DATA_DIR / "EN_human_annotations.csv", DATA_DIR / "EN_LLM_annotations.csv"),
	"IT": (DATA_DIR / "IT_human_annotations.csv", DATA_DIR / "IT_LLM_annotations.csv"),
	"SI": (DATA_DIR / "SI_human_annotations.csv", DATA_DIR / "SI_LLM_annotations.csv"),
}


LABEL_COLUMNS = ("label_run1", "label_run2")
LABEL_IDS = (0, 1, 2, 3)


def load_labels(path: Path) -> pd.Series:
	frame = pd.read_csv(path, usecols=list(LABEL_COLUMNS), low_memory=False)
	values = pd.concat([frame[column] for column in LABEL_COLUMNS], ignore_index=True)
	values = pd.to_numeric(values, errors="coerce").dropna().astype(int)
	values = values[values.isin(LABEL_IDS)]
	return values


def summarize_group(name: str, paths: list[Path]) -> None:
	labels = pd.concat([load_labels(path) for path in paths], ignore_index=True)
	total = len(labels)
	counts = labels.value_counts().reindex(LABEL_IDS, fill_value=0).sort_index()

	print(f"{name} annotations")
	print(f"  total labels: {total}")
	for label_id in LABEL_IDS:
		count = int(counts.loc[label_id])
		share = (count / total * 100) if total else 0.0
		print(f"  label {label_id}: {count} ({share:.2f}%)")
	print()


def compute_pairwise_alpha(human_path: Path, llm_path: Path) -> float:
	alpha_values: list[float] = []
	for column in LABEL_COLUMNS:
		human_frame = pd.read_csv(human_path, usecols=[column], low_memory=False)
		llm_frame = pd.read_csv(llm_path, usecols=[column], low_memory=False)

		human_values = pd.to_numeric(human_frame[column], errors="coerce")
		llm_values = pd.to_numeric(llm_frame[column], errors="coerce")
		valid_mask = (
			human_values.notna()
			& llm_values.notna()
			& human_values.isin(LABEL_IDS)
			& llm_values.isin(LABEL_IDS)
		)
		if valid_mask.sum() < 2:
			continue

		alpha_values.append(
			float(
				krippendorff.alpha(
					reliability_data=np.vstack(
						[
							human_values[valid_mask].astype(int).to_numpy(),
							llm_values[valid_mask].astype(int).to_numpy(),
						]
					),
					level_of_measurement="ordinal",
				)
			)
		)

	if not alpha_values:
		return float("nan")
	return float(np.mean(alpha_values))


def print_pairwise_alpha(language: str, human_path: Path, llm_path: Path) -> None:
	alpha = compute_pairwise_alpha(human_path, llm_path)
	print(f"{language} human vs LLM agreement")
	print(f"  krippendorff ordinal alpha: {alpha:.4f}")
	print()


def main() -> None:
	for name, paths in SOURCE_GROUPS.items():
		summarize_group(name, paths)

	for language, (human_path, llm_path) in PAIRWISE_LANGUAGE_PAIRS.items():
		print_pairwise_alpha(language, human_path, llm_path)


if __name__ == "__main__":
	main()
"""Compare one language's human and LLM annotation CSVs by agreement metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

try:
	import krippendorff as krippendorff_lib
except ImportError:  # pragma: no cover - optional dependency fallback
	krippendorff_lib = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
LABELS = [0, 1, 2, 3]
LANGUAGE = "SI"
VALID_LANGUAGES = {"EN", "IT", "SI"}


def read_annotations(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing file: {path}")
	return pd.read_csv(path)


def to_numeric_series(values: pd.Series) -> np.ndarray:
	return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def ordinal_distance(left: int, right: int, counts: pd.Series) -> float:
	if left == right:
		return 0.0
	low = min(left, right)
	high = max(left, right)
	between = counts.loc[(counts.index >= low) & (counts.index < high)].sum()
	total = float(counts.sum())
	if total == 0:
		return 0.0
	return float((between / total) ** 2)


def fallback_krippendorff_ordinal_alpha(data: np.ndarray) -> float:
	if data.size == 0 or data.ndim != 2 or data.shape[0] < 2:
		return float("nan")

	valid_values = data[~np.isnan(data)]
	if valid_values.size < 2:
		return float("nan")

	categories = sorted({int(value) for value in valid_values})
	counts = pd.Series(valid_values.astype(int)).value_counts().reindex(categories, fill_value=0).sort_index()

	observed_numerator = 0.0
	observed_denominator = 0
	for item_index in range(data.shape[1]):
		item_labels = [int(value) for value in data[:, item_index] if not np.isnan(value)]
		if len(item_labels) < 2:
			continue
		for left_index, left in enumerate(item_labels[:-1]):
			for right in item_labels[left_index + 1:]:
				observed_numerator += ordinal_distance(left, right, counts)
				observed_denominator += 1

	if observed_denominator == 0:
		return float("nan")

	expected_numerator = 0.0
	expected_denominator = 0
	for left in categories:
		for right in categories:
			if left >= right:
				continue
			pair_count = int(counts.loc[left]) * int(counts.loc[right])
			expected_numerator += pair_count * ordinal_distance(left, right, counts)
			expected_denominator += pair_count

	if expected_denominator == 0:
		return 1.0

	observed = observed_numerator / observed_denominator
	expected = expected_numerator / expected_denominator
	if expected == 0:
		return 1.0
	return 1.0 - (observed / expected)


def krippendorff_ordinal_alpha(columns: Iterable[np.ndarray]) -> float:
	data = np.array(list(columns), dtype=float)
	if data.ndim != 2 or data.shape[0] < 2:
		return float("nan")
	if krippendorff_lib is not None:
		return float(krippendorff_lib.alpha(data, level_of_measurement="ordinal"))
	return fallback_krippendorff_ordinal_alpha(data)


def summarize_pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
	valid_mask = ~np.isnan(reference) & ~np.isnan(prediction)
	if not valid_mask.any():
		return float("nan"), float("nan")
	ref_values = reference[valid_mask].astype(int)
	pred_values = prediction[valid_mask].astype(int)
	macro_f1 = f1_score(ref_values, pred_values, average="macro", zero_division=0)
	alpha = krippendorff_ordinal_alpha([reference[valid_mask], prediction[valid_mask]])
	return alpha, macro_f1


def annotation_paths_for(language: str) -> tuple[Path, Path]:
	human_candidates = [
		DATA_DIR / f"{language}_human_annotations.csv",
		DATA_DIR / f"{language}_human_annotation.csv",
	]
	llm_candidates = [
		DATA_DIR / f"{language}_LLM_annotations.csv",
		DATA_DIR / f"{language}_LLM_annotation.csv",
	]
	human_path = next((path for path in human_candidates if path.exists()), None)
	llm_path = next((path for path in llm_candidates if path.exists()), None)
	if human_path is None:
		raise FileNotFoundError(f"Could not find a human annotation file for {language}")
	if llm_path is None:
		raise FileNotFoundError(f"Could not find an LLM annotation file for {language}")
	return human_path, llm_path


def compare_pair(language: str, human_path: Path, llm_path: Path) -> None:
	human = read_annotations(human_path)
	llm = read_annotations(llm_path)

	required_columns = {"index", "text", "label_run1", "label_run2"}
	if not required_columns.issubset(human.columns):
		raise ValueError(f"{human_path.name} must contain {sorted(required_columns)}")
	if not required_columns.issubset(llm.columns):
		raise ValueError(f"{llm_path.name} must contain {sorted(required_columns)}")

	merged = human.merge(llm, on="index", suffixes=("_human", "_llm"), how="inner")
	if merged.empty:
		raise ValueError(f"No overlapping rows found between {human_path.name} and {llm_path.name}")

	text_mismatches = merged[merged["text_human"] != merged["text_llm"]]
	if not text_mismatches.empty:
		print(f"[WARNING] {language}: {len(text_mismatches)} text mismatches after index merge.")

	human_run1 = to_numeric_series(merged["label_run1_human"])
	human_run2 = to_numeric_series(merged["label_run2_human"])
	llm_run1 = to_numeric_series(merged["label_run1_llm"])
	llm_run2 = to_numeric_series(merged["label_run2_llm"])

	human_within_alpha = krippendorff_ordinal_alpha([human_run1, human_run2])
	llm_within_alpha = krippendorff_ordinal_alpha([llm_run1, llm_run2])

	flat_human = np.concatenate([human_run1, human_run2])
	flat_llm = np.concatenate([llm_run1, llm_run2])
	valid_flat_mask = ~np.isnan(flat_human) & ~np.isnan(flat_llm)
	flat_human_valid = flat_human[valid_flat_mask].astype(int)
	flat_llm_valid = flat_llm[valid_flat_mask].astype(int)
	conf_matrix = confusion_matrix(flat_human_valid, flat_llm_valid, labels=LABELS)
	overall_macro_f1 = f1_score(flat_human_valid, flat_llm_valid, average="macro", zero_division=0)
	overall_accuracy = float(np.mean(flat_human_valid == flat_llm_valid)) if len(flat_human_valid) else float("nan")

	print(f"{language}: {human_path.name} vs {llm_path.name}")
	print(f"  paired labels used: {len(flat_human_valid)}")
	print(f"  macro F1 (all labels): {overall_macro_f1:.4f}")
	print(f"  accuracy vs human labels: {overall_accuracy:.4f}")
	print(f"  alpha within LLM labels: {llm_within_alpha:.4f}")
	print(f"  alpha within human labels: {human_within_alpha:.4f}")
	print("  confusion matrix (rows=human, cols=LLM)")
	confusion_df = pd.DataFrame(conf_matrix, index=LABELS, columns=LABELS)
	print(confusion_df.to_string())
	print()


def main() -> None:
	language = LANGUAGE.strip().upper()
	if language not in VALID_LANGUAGES:
		raise ValueError("LANGUAGE must be one of: EN, IT, SI")
	human_path, llm_path = annotation_paths_for(language)
	compare_pair(language, human_path, llm_path)


if __name__ == "__main__":
	main()

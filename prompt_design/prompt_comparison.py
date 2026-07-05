"""Compare prompt outputs against Sample150 labels by agreement metrics.

The script reports the requested metrics for the prompt variants
`redefined_distinction` and `biased_personas` across EN, IT, and SI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
	import krippendorff as krippendorff_lib
except ImportError:  # pragma: no cover - optional dependency fallback
	krippendorff_lib = None


BASE_DIR = Path(__file__).resolve().parent

LANGUAGE_CONFIG = {
	"EN": {
		"sample_path": BASE_DIR / "Data" / "Sample150_EN.csv",
		"prompt_dir": BASE_DIR / "prompt_design" / "EN",
		"sample_first_col": "type_1",
		"sample_second_col": "type_2",
		"prompt_first_col": "label_run1",
		"prompt_second_col": "label_run2",
	},
	"IT": {
		"sample_path": BASE_DIR / "Data" / "Sample150_IT.csv",
		"prompt_dir": BASE_DIR / "prompt_design" / "IT",
		"sample_first_col": "type_1",
		"sample_second_col": "type_2",
		"prompt_first_col": "label_run1",
		"prompt_second_col": "label_run2",
	},
	"SI": {
		"sample_path": BASE_DIR / "Data" / "Sample150_SI.csv",
		"prompt_dir": BASE_DIR / "prompt_design" / "SI",
		"sample_first_col": "type_1",
		"sample_second_col": "type_2",
		"prompt_first_col": "label_run1",
		"prompt_second_col": "label_run2",
	},
}


PROMPT_VARIANTS = [
	"redefined_distinction",
	"biased_personas",
]


def read_labels(path: Path) -> pd.DataFrame:
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
	distance = (between / total) ** 2
	return float(distance)


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


def align_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
	if {"text", "comment"}.issubset(left.columns) and {"text", "comment"}.issubset(right.columns):
		merged = left.merge(right, left_on="text", right_on="comment", how="inner", suffixes=("_left", "_right"))
		if not merged.empty:
			return merged

	if len(left) != len(right):
		raise ValueError("Files do not share a merge key and row counts differ, so alignment is ambiguous.")

	merged = left.reset_index(drop=True).copy()
	right_reset = right.reset_index(drop=True).copy()
	for column in right_reset.columns:
		if column in merged.columns:
			merged[f"{column}_right"] = right_reset[column]
		else:
			merged[column] = right_reset[column]
	return merged


def compare_pair(
	language: str,
	sample_path: Path,
	prompt_path: Path,
	sample_first_col: str,
	sample_second_col: str,
	prompt_first_col: str,
	prompt_second_col: str,
) -> dict[str, float | str]:
	sample = read_labels(sample_path)
	prompt = read_labels(prompt_path)
	merged = align_frames(sample, prompt)

	matched_rows = len(merged)

	sample_first = to_numeric_series(merged[sample_first_col])
	sample_second = to_numeric_series(merged[sample_second_col])
	prompt_first = to_numeric_series(merged[prompt_first_col])
	prompt_second = to_numeric_series(merged[prompt_second_col])

	def summarize_pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
		valid_mask = ~np.isnan(reference) & ~np.isnan(prediction)
		if not valid_mask.any():
			return float("nan"), float("nan"), float("nan")
		ref_values = reference[valid_mask].astype(int)
		pred_values = prediction[valid_mask].astype(int)
		alpha = krippendorff_ordinal_alpha([reference[valid_mask], prediction[valid_mask]])
		accuracy = accuracy_score(ref_values, pred_values)
		macro_f1 = f1_score(ref_values, pred_values, average="macro", zero_division=0)
		return alpha, accuracy, macro_f1

	within_sample_alpha = krippendorff_ordinal_alpha([sample_first, sample_second])
	within_prompt_alpha = krippendorff_ordinal_alpha([prompt_first, prompt_second])
	between_first_alpha, between_first_accuracy, between_first_macro_f1 = summarize_pair(sample_first, prompt_first)
	between_second_alpha, between_second_accuracy, between_second_macro_f1 = summarize_pair(sample_second, prompt_second)
	type1_vs_run2_alpha, type1_vs_run2_accuracy, type1_vs_run2_macro_f1 = summarize_pair(sample_first, prompt_second)
	type2_vs_run1_alpha, type2_vs_run1_accuracy, type2_vs_run1_macro_f1 = summarize_pair(sample_second, prompt_first)
	type2_vs_run2_alpha, type2_vs_run2_accuracy, type2_vs_run2_macro_f1 = summarize_pair(sample_second, prompt_second)
	macro_f1_values = [
		between_first_macro_f1,
		type2_vs_run1_macro_f1,
		type1_vs_run2_macro_f1,
		type2_vs_run2_macro_f1,
	]
	mean_macro_f1 = float(np.nanmean(macro_f1_values))

	return {
		"language": language,
		"prompt_file": prompt_path.name,
		"matched_rows": float(matched_rows),
		"sample_alpha": within_sample_alpha,
		"prompt_alpha": within_prompt_alpha,
		"between_alpha": between_first_alpha,
		"between_accuracy": between_first_accuracy,
		"between_macro_f1": between_first_macro_f1,
		"type2_vs_run1_alpha": type2_vs_run1_alpha,
		"type1_vs_run2_alpha": type1_vs_run2_alpha,
		"type2_vs_run2_alpha": type2_vs_run2_alpha,
		"mean_macro_f1": mean_macro_f1,
	}


def summarize_sample_baseline(sample_path: Path, sample_first_col: str, sample_second_col: str) -> dict[str, float | str]:
	sample = read_labels(sample_path)

	sample_first = to_numeric_series(sample[sample_first_col])
	sample_second = to_numeric_series(sample[sample_second_col])
	valid_mask = ~np.isnan(sample_first) & ~np.isnan(sample_second)
	if not valid_mask.any():
		return {
			"sample_file": sample_path.name,
			"sample_alpha": float("nan"),
			"sample_accuracy": float("nan"),
			"sample_macro_f1": float("nan"),
		}

	ref_values = sample_first[valid_mask].astype(int)
	pred_values = sample_second[valid_mask].astype(int)
	return {
		"sample_file": sample_path.name,
		"sample_alpha": krippendorff_ordinal_alpha([sample_first[valid_mask], sample_second[valid_mask]]),
		"sample_accuracy": accuracy_score(ref_values, pred_values),
		"sample_macro_f1": f1_score(ref_values, pred_values, average="macro", zero_division=0),
	}


def main() -> None:
	results: list[dict[str, float | str]] = []
	for language in ("EN", "IT", "SI"):
		config = LANGUAGE_CONFIG[language]
		sample_metrics = summarize_sample_baseline(
			config["sample_path"],
			config["sample_first_col"],
			config["sample_second_col"],
		)
		for prompt_variant in PROMPT_VARIANTS:
			prompt_path = config["prompt_dir"] / f"{language}_prompt_{prompt_variant}.csv"
			results.append(
				compare_pair(
					language,
					config["sample_path"],
					prompt_path,
					config["sample_first_col"],
					config["sample_second_col"],
					config["prompt_first_col"],
					config["prompt_second_col"],
				)
			)
			results[-1]["sample_file"] = sample_metrics["sample_file"]
			results[-1]["sample_alpha"] = sample_metrics["sample_alpha"]

	print("language | prompt file | mean macro F1 | between alpha | prompt alpha | matched rows | sample alpha")
	for item in results:
		print(
			f"{item['language']} | {item['prompt_file']} | {item['mean_macro_f1']:.4f} | "
			f"{item['between_alpha']:.4f} | {item['prompt_alpha']:.4f} | "
			f"{int(item['matched_rows'])} | {item['sample_alpha']:.4f}"
		)


if __name__ == "__main__":
	main()

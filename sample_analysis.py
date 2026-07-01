from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

SAMPLED_FILES = {
	"EN": DATA_DIR / "Sample150_EN.csv",
	"IT": DATA_DIR / "Sample150_IT.csv",
	"SI": DATA_DIR / "Sample150_SI.csv",
}


def print_type_distribution(sample_path: Path, label: str) -> None:
	dataframe = pd.read_csv(sample_path)
	if "Type" not in dataframe.columns:
		raise KeyError(f"Missing Type column in {sample_path.name}")

	counts = dataframe["Type"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
	present_types = int((counts > 0).sum())
	total_rows = len(dataframe)

	print(f"{label}: {sample_path.name}")
	print(f"  Different types present: {present_types}")
	for type_value, count in counts.items():
		percentage = (count / total_rows * 100) if total_rows else 0
		print(f"  Type {type_value}: {count} ({percentage:.1f}%)")
	print()


def main() -> None:
	for label, sample_path in SAMPLED_FILES.items():
		print_type_distribution(sample_path, label)


if __name__ == "__main__":
	main()

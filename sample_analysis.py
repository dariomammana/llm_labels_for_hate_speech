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
	
	# Check if file has type_1 and type_2 columns
	if "type_1" in dataframe.columns and "type_2" in dataframe.columns:
		total_rows = len(dataframe)
		
		print(f"{label}: {sample_path.name}")
		print(f"  Total annotations: {total_rows * 2}")
		print()
		
		# Analyze type_1
		counts_type1 = dataframe["type_1"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
		present_types_1 = int((counts_type1 > 0).sum())
		print(f"  Type_1 Distribution:")
		print(f"    Different types present: {present_types_1}")
		for type_value, count in counts_type1.items():
			percentage = (count / total_rows * 100) if total_rows else 0
			print(f"    Type {type_value}: {count} ({percentage:.1f}%)")
		print()
		
		# Analyze type_2
		counts_type2 = dataframe["type_2"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
		present_types_2 = int((counts_type2 > 0).sum())
		print(f"  Type_2 Distribution:")
		print(f"    Different types present: {present_types_2}")
		for type_value, count in counts_type2.items():
			percentage = (count / total_rows * 100) if total_rows else 0
			print(f"    Type {type_value}: {count} ({percentage:.1f}%)")
		print()
		
		# Combined distribution (both annotators together)
		all_types = pd.concat([dataframe["type_1"], dataframe["type_2"]])
		counts_combined = all_types.value_counts().reindex([0, 1, 2, 3], fill_value=0)
		print(f"  Combined Distribution (Type_1 + Type_2):")
		for type_value, count in counts_combined.items():
			percentage = (count / (total_rows * 2) * 100) if total_rows else 0
			print(f"    Type {type_value}: {count} ({percentage:.1f}%)")
		print()
		
	elif "type" in dataframe.columns:
		counts = dataframe["type"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
		present_types = int((counts > 0).sum())
		total_rows = len(dataframe)

		print(f"{label}: {sample_path.name}")
		print(f"  Different types present: {present_types}")
		for type_value, count in counts.items():
			percentage = (count / total_rows * 100) if total_rows else 0
			print(f"  Type {type_value}: {count} ({percentage:.1f}%)")
		print()
	else:
		raise KeyError(f"Missing type columns in {sample_path.name}")


def main() -> None:
	for label, sample_path in SAMPLED_FILES.items():
		print_type_distribution(sample_path, label)


if __name__ == "__main__":
	main()

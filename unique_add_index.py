from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"


def add_index_column(csv_path: Path) -> bool:
	dataframe = pd.read_csv(csv_path)
	if "index" in dataframe.columns:
		return False

	dataframe.insert(0, "index", range(1, len(dataframe) + 1))
	dataframe.to_csv(csv_path, index=False)
	return True


def main() -> None:
	unique_files = sorted(DATA_DIR.glob("Unique_*.csv"))
	if not unique_files:
		print("No Unique_*.csv files found.")
		return

	for csv_path in unique_files:
		updated = add_index_column(csv_path)
		status = "updated" if updated else "skipped (already has index column)"
		print(f"{csv_path.name}: {status}")


if __name__ == "__main__":
	main()

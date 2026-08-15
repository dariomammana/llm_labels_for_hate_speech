from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
TEST_DATA_DIR = DATA_DIR / "Test_data"


# Direct downloads from CLARIN.SI handles requested by the project owner.
DOWNLOAD_SPECS = [
    {
        "description": "EN train",
        "url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1454/IMSyPP_EN_YouTube_comments_train.csv?sequence=1&isAllowed=y",
        "destination": DATA_DIR / "IMSyPP_EN_YouTube_comments_train.csv",
    },
    {
        "description": "EN evaluation no-context",
        "url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1454/IMSyPP_EN_YouTube_comments_evaluation_no_context.csv?sequence=3&isAllowed=y",
        "destination": TEST_DATA_DIR / "IMSyPP_EN_YouTube_comments_evaluation_no_context.csv",
    },
    {
        "description": "IT train",
        "url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1450/IMSyPP_IT_YouTube_comments_train.csv?sequence=2&isAllowed=y",
        "destination": DATA_DIR / "IMSyPP_IT_YouTube_comments_train.csv",
    },
    {
        "description": "IT evaluation",
        "url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1450/IMSyPP_IT_YouTube_comments_evaluation.csv?sequence=1&isAllowed=y",
        "destination": TEST_DATA_DIR / "IMSyPP_IT_YouTube_comments_evaluation.csv",
    },
    {
        "description": "SI train (normalized filename for this repo)",
        "url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1398/IMSyPP_SI_anotacije_training-clarin.csv?sequence=6&isAllowed=y",
        "destination": DATA_DIR / "IMSyPP_SI_anotacije_round1(in).csv",
    },
    {
        "description": "SI evaluation (normalized filename for this repo)",
        "url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1398/IMSyPP_SI_anotacije_evaluation-clarin.csv?sequence=8&isAllowed=y",
        "destination": TEST_DATA_DIR / "IMSyPP_SI_anotacije_round2.csv",
    },
]


def download_file(url: str, destination: Path, force: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        return "skipped"

    # Write to a temp file first, then move atomically to reduce risk of partial files.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        with urlopen(url) as response, tmp_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
        shutil.move(str(tmp_path), str(destination))
    except URLError as error:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {error}") from error
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download required IMSyPP CSV datasets from CLARIN.SI into this repository."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Preparing dataset directories...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CLARIN datasets:")
    failures = []
    for spec in DOWNLOAD_SPECS:
        try:
            status = download_file(spec["url"], spec["destination"], force=args.force)
            print(f"  [{status.upper()}] {spec['description']}: {spec['destination']}")
        except Exception as error:
            failures.append((spec, error))
            print(f"  [FAILED] {spec['description']}: {error}")

    if failures:
        print("\nOne or more downloads failed.")
        return 1

    print("\nAll required CSV sources are present.")
    print("Next steps:")
    print("  1) python human_labels_import.py")
    print("  2) python unique_add_index.py")
    print("  3) python test_set_cleanup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

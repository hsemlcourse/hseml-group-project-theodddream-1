"""Data acquisition: download dataset from Kaggle, extract, validate schema, enrich with geo-data."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_SLUG = "shivamb/bank-customer-segmentation"
CSV_NAME = "bank_transactions.csv"
ZIP_NAME = "bank_transactions.csv.zip"

EXPECTED_COLUMNS = [
    "TransactionID",
    "CustomerID",
    "CustomerDOB",
    "CustGender",
    "CustLocation",
    "CustAccountBalance",
    "TransactionDate",
    "TransactionTime",
    "TransactionAmount (INR)",
]

INDIA_CITIES_URL = (
    "https://raw.githubusercontent.com/nshntarora/Indian-Cities-JSON/master/cities.json"
)


def download_from_kaggle(dest_dir: Path | None = None) -> Path:
    """Download dataset via Kaggle CLI (requires ~/.kaggle/kaggle.json configured).

    Returns path to the extracted CSV.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Also configure ~/.kaggle/kaggle.json with your API token."
        ) from e

    dest_dir = dest_dir or RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(dest_dir), unzip=True)

    csv_path = dest_dir / CSV_NAME
    if not csv_path.exists():
        for f in dest_dir.iterdir():
            if f.suffix == ".csv":
                csv_path = f
                break
    return csv_path


def extract_from_zip(zip_path: Path | None = None, dest_dir: Path | None = None) -> Path:
    """Extract CSV from a local zip file."""
    zip_path = zip_path or (RAW_DIR / ZIP_NAME)
    dest_dir = dest_dir or RAW_DIR

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV files found in {zip_path}")
        zf.extract(csv_members[0], path=str(dest_dir))
        return dest_dir / csv_members[0]


def validate_schema(csv_path: Path) -> dict:
    """Validate that the CSV matches expected schema. Returns summary stats."""
    df = pd.read_csv(csv_path, nrows=5)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df_full = pd.read_csv(csv_path)
    return {
        "path": str(csv_path),
        "rows": len(df_full),
        "columns": list(df_full.columns),
        "extra_columns": list(extra_cols),
        "dtypes": df_full.dtypes.astype(str).to_dict(),
        "null_counts": df_full.isnull().sum().to_dict(),
        "valid": True,
    }


def fetch_india_cities_geo() -> pd.DataFrame:
    """Parse Indian cities with state info from a public JSON source.

    Returns DataFrame with columns: city, state (useful for geo-enrichment of CustLocation).
    """
    req = Request(INDIA_CITIES_URL, headers={"User-Agent": "Python/hseml-project"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    records = []
    for item in data:
        city = item.get("name", "").strip().upper()
        state = item.get("state", "").strip()
        if city:
            records.append({"city": city, "state": state})

    return pd.DataFrame(records).drop_duplicates(subset=["city"])


def enrich_with_geo(df: pd.DataFrame, geo_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add state information to transactions based on CustLocation.

    Returns (enriched_df, geo_df_used).
    """
    if geo_df is None:
        geo_df = fetch_india_cities_geo()

    location_upper = df["CustLocation"].str.strip().str.upper()
    state_map = geo_df.set_index("city")["state"].to_dict()
    df = df.copy()
    df["CustState"] = location_upper.map(state_map).fillna("UNKNOWN")
    return df, geo_df


def main():
    """CLI entry point: download/extract data, validate, fetch geo info."""
    import argparse

    parser = argparse.ArgumentParser(description="Data acquisition and validation pipeline")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle API")
    parser.add_argument("--extract", action="store_true", help="Extract from local ZIP")
    parser.add_argument("--validate", action="store_true", help="Validate CSV schema")
    parser.add_argument("--geo", action="store_true", help="Fetch geo-enrichment data")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    args = parser.parse_args()

    if args.all or args.download:
        print("[1/4] Downloading from Kaggle...")
        csv_path = download_from_kaggle()
        print(f"  -> {csv_path}")
    elif args.all or args.extract:
        print("[1/4] Extracting from ZIP...")
        csv_path = extract_from_zip()
        print(f"  -> {csv_path}")
    else:
        csv_path = RAW_DIR / CSV_NAME

    if args.all or args.validate:
        print("[2/4] Validating schema...")
        report = validate_schema(csv_path)
        print(f"  -> {report['rows']} rows, {len(report['columns'])} columns, valid={report['valid']}")

    if args.all or args.geo:
        print("[3/4] Fetching geo data...")
        geo_df = fetch_india_cities_geo()
        geo_path = RAW_DIR / "india_cities_geo.csv"
        geo_df.to_csv(geo_path, index=False)
        print(f"  -> {len(geo_df)} cities saved to {geo_path}")

    print("[4/4] Done.")


if __name__ == "__main__":
    main()

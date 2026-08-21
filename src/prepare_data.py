from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# prepare_data.py נמצא בתוך src,
# ולכן parent.parent הוא השורש של הפרויקט.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

ABC_FILE = RAW_DATA_DIR / "abcnews-date-text.csv"
BRENT_FILE = RAW_DATA_DIR / "rbrted.xls"

ABC_DAILY_OUTPUT = PROCESSED_DATA_DIR / "abc_daily_counts.csv"
BRENT_OUTPUT = PROCESSED_DATA_DIR / "brent_clean.csv"
MERGED_OUTPUT = PROCESSED_DATA_DIR / "abc_brent_daily.csv"


def check_file_exists(file_path: Path) -> None:
    """
    Verifies that a required input file exists.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the file:\n{file_path}\n\n"
            "Make sure it is located inside data/raw."
        )


def prepare_abc_news(file_path: Path) -> pd.DataFrame:
    """
    Loads and cleans the ABC News dataset.

    Returns a daily table with:
    - date
    - total_headlines
    """
    print("\nLoading ABC News data...")

    news = pd.read_csv(
        file_path,
        usecols=["publish_date", "headline_text"],
        dtype={
            "publish_date": "string",
            "headline_text": "string",
        },
        low_memory=False,
    )

    original_row_count = len(news)

    # Convert dates from values such as 20030219 to 2003-02-19.
    news["date"] = pd.to_datetime(
        news["publish_date"],
        format="%Y%m%d",
        errors="coerce",
    )

    # Remove unnecessary spaces from headlines.
    news["headline_text"] = news["headline_text"].str.strip()

    invalid_dates = news["date"].isna().sum()
    missing_headlines = news["headline_text"].isna().sum()

    # Remove rows without a valid date or headline.
    news = news.dropna(subset=["date", "headline_text"])

    # Remove headlines that became empty after stripping whitespace.
    news = news[news["headline_text"] != ""]

    rows_before_duplicates = len(news)

    # The same headline may legitimately appear on different dates,
    # so duplicates are removed only when both date and headline match.
    news = news.drop_duplicates(
        subset=["date", "headline_text"],
        keep="first",
    )

    duplicate_count = rows_before_duplicates - len(news)

    # Count all news headlines published on each date.
    daily_news = (
        news.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "total_headlines"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    print("ABC News cleaning completed.")
    print(f"Original number of rows: {original_row_count:,}")
    print(f"Invalid dates removed: {invalid_dates:,}")
    print(f"Missing headlines removed: {missing_headlines:,}")
    print(f"Duplicate date-headline rows removed: {duplicate_count:,}")
    print(f"Remaining unique headlines: {len(news):,}")
    print(f"Number of dates with headlines: {len(daily_news):,}")
    print(
        "ABC date range: "
        f"{daily_news['date'].min().date()} "
        f"to {daily_news['date'].max().date()}"
    )

    return daily_news


def prepare_brent_prices(file_path: Path) -> pd.DataFrame:
    """
    Loads and cleans daily Brent oil prices from the EIA Excel file.

    Returns a table with:
    - date
    - brent_price
    """
    print("\nLoading Brent price data...")

    try:
        # The actual data is in the sheet named "Data 1".
        # The first two rows contain metadata, and row 3 contains headers.
        brent = pd.read_excel(
            file_path,
            sheet_name="Data 1",
            header=2,
            usecols="A:B",
            engine="xlrd",
        )
    except ImportError as error:
        raise ImportError(
            "Reading .xls files requires the xlrd package.\n"
            "Run this command in the PyCharm terminal:\n"
            "pip install xlrd"
        ) from error

    if brent.shape[1] != 2:
        raise ValueError(
            "The Brent file does not contain the expected two columns."
        )

    # Rename the long original EIA column names.
    brent.columns = ["date", "brent_price"]

    original_row_count = len(brent)

    brent["date"] = pd.to_datetime(
        brent["date"],
        errors="coerce",
    )

    brent["brent_price"] = pd.to_numeric(
        brent["brent_price"],
        errors="coerce",
    )

    invalid_dates = brent["date"].isna().sum()
    invalid_prices = brent["brent_price"].isna().sum()

    # Remove rows without a valid date or price.
    brent = brent.dropna(subset=["date", "brent_price"])

    rows_before_duplicates = len(brent)

    # There should be one Brent value per date.
    brent = brent.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    duplicate_count = rows_before_duplicates - len(brent)

    brent = (
        brent.sort_values("date")
        .reset_index(drop=True)
    )

    print("Brent price cleaning completed.")
    print(f"Original number of rows: {original_row_count:,}")
    print(f"Invalid dates removed: {invalid_dates:,}")
    print(f"Invalid prices removed: {invalid_prices:,}")
    print(f"Duplicate dates removed: {duplicate_count:,}")
    print(f"Remaining Brent records: {len(brent):,}")
    print(
        "Brent date range: "
        f"{brent['date'].min().date()} "
        f"to {brent['date'].max().date()}"
    )

    return brent


def merge_daily_data(
    daily_news: pd.DataFrame,
    brent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges news counts and Brent prices by date.

    The inner join keeps dates for which both:
    - a Brent price exists;
    - ABC headlines exist.
    """
    print("\nMerging the datasets by date...")

    overlap_start = max(
        daily_news["date"].min(),
        brent["date"].min(),
    )

    overlap_end = min(
        daily_news["date"].max(),
        brent["date"].max(),
    )

    merged = pd.merge(
        brent,
        daily_news,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged.sort_values("date")
        .reset_index(drop=True)
    )

    print(f"Common date range: {overlap_start.date()} to {overlap_end.date()}")
    print(f"Number of matched dates: {len(merged):,}")

    return merged


def save_data(
    daily_news: pd.DataFrame,
    brent: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    """
    Saves all processed datasets.
    """
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_news.to_csv(
        ABC_DAILY_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    brent.to_csv(
        BRENT_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    merged.to_csv(
        MERGED_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    print("\nSaved processed files:")
    print(f"1. {ABC_DAILY_OUTPUT}")
    print(f"2. {BRENT_OUTPUT}")
    print(f"3. {MERGED_OUTPUT}")


def main() -> None:
    """
    Main project data-preparation pipeline.
    """
    print("=" * 60)
    print("GeoOil-Pulse: Initial Data Preparation")
    print("=" * 60)

    check_file_exists(ABC_FILE)
    check_file_exists(BRENT_FILE)

    daily_news = prepare_abc_news(ABC_FILE)
    brent = prepare_brent_prices(BRENT_FILE)

    merged = merge_daily_data(
        daily_news=daily_news,
        brent=brent,
    )

    save_data(
        daily_news=daily_news,
        brent=brent,
        merged=merged,
    )

    print("\nFirst five merged rows:")
    print(merged.head().to_string(index=False))

    print("\nLast five merged rows:")
    print(merged.tail().to_string(index=False))

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()
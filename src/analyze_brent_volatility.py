from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

INPUT_FILE = PROCESSED_DATA_DIR / "abc_brent_oil_daily.csv"

FEATURES_OUTPUT = PROCESSED_DATA_DIR / "brent_market_features.csv"
EXTREME_DAYS_OUTPUT = PROCESSED_DATA_DIR / "brent_extreme_days_p95.csv"
YEARLY_SUMMARY_OUTPUT = PROCESSED_DATA_DIR / "brent_extreme_days_by_year.csv"

PRICE_PLOT_OUTPUT = OUTPUT_DIR / "brent_price_over_time.png"
RETURNS_PLOT_OUTPUT = OUTPUT_DIR / "brent_daily_returns.png"
VOLATILITY_PLOT_OUTPUT = OUTPUT_DIR / "brent_rolling_volatility.png"
EXTREME_PLOT_OUTPUT = OUTPUT_DIR / "brent_extreme_price_days.png"

ROLLING_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252
EXTREME_PERCENTILES = (0.90, 0.95, 0.99)


# ---------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------

def check_file_exists(file_path: Path) -> None:
    """Verifies that the required input file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the file:\n{file_path}\n\n"
            "Run prepare_data.py and filter_oil_headlines.py first."
        )


def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the combined ABC News and Brent dataset."""
    print("\nLoading combined Brent and news data...")

    data = pd.read_csv(
        file_path,
        parse_dates=["date"],
    )

    required_columns = {"date", "brent_price"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data["brent_price"] = pd.to_numeric(
        data["brent_price"],
        errors="coerce",
    )

    invalid_rows = data[["date", "brent_price"]].isna().any(axis=1)
    invalid_count = int(invalid_rows.sum())

    if invalid_count > 0:
        print(f"Removing invalid date/price rows: {invalid_count:,}")
        data = data.loc[~invalid_rows].copy()

    duplicate_dates = int(data["date"].duplicated().sum())

    if duplicate_dates > 0:
        raise ValueError(
            f"Found {duplicate_dates:,} duplicate dates in the input file."
        )

    non_positive_prices = int((data["brent_price"] <= 0).sum())

    if non_positive_prices > 0:
        raise ValueError(
            "Brent prices must be positive before return calculations. "
            f"Found {non_positive_prices:,} non-positive values."
        )

    data = (
        data.sort_values("date")
        .reset_index(drop=True)
    )

    print(f"Rows loaded: {len(data):,}")
    print(
        "Date range: "
        f"{data['date'].min().date()} to {data['date'].max().date()}"
    )

    return data


# ---------------------------------------------------------
# Market features
# ---------------------------------------------------------

def add_return_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds daily price-change and volatility variables.

    daily_return is a decimal value:
    0.04 means a 4% increase.
    """
    print("\nCalculating daily returns and rolling volatility...")

    features = data.copy()

    features["price_change"] = features["brent_price"].diff()

    features["daily_return"] = features["brent_price"].pct_change(
        fill_method=None
    )

    features["daily_return_pct"] = features["daily_return"] * 100

    features["absolute_return"] = features["daily_return"].abs()
    features["absolute_return_pct"] = features["absolute_return"] * 100

    # Log returns are included as an additional standard market measure.
    features["log_return"] = np.log(
        features["brent_price"]
        / features["brent_price"].shift(1)
    )

    features[f"rolling_volatility_{ROLLING_WINDOW}d"] = (
        features["daily_return"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        )
        .std(ddof=1)
    )

    features[f"rolling_volatility_{ROLLING_WINDOW}d_pct"] = (
        features[f"rolling_volatility_{ROLLING_WINDOW}d"] * 100
    )

    # Annualized volatility is useful for interpretation and comparison.
    features[f"annualized_volatility_{ROLLING_WINDOW}d"] = (
        features[f"rolling_volatility_{ROLLING_WINDOW}d"]
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    features[f"annualized_volatility_{ROLLING_WINDOW}d_pct"] = (
        features[f"annualized_volatility_{ROLLING_WINDOW}d"] * 100
    )

    features["return_direction"] = np.select(
        [
            features["daily_return"] > 0,
            features["daily_return"] < 0,
        ],
        ["increase", "decrease"],
        default="no_change",
    )

    # The first record has no previous trading day.
    features.loc[
        features["daily_return"].isna(),
        "return_direction",
    ] = "not_available"

    return features


def add_extreme_day_flags(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[float, float]]:
    """
    Marks extreme days using the 90th, 95th and 99th percentiles
    of the absolute daily return distribution.
    """
    print("\nCalculating percentile thresholds for extreme days...")

    valid_absolute_returns = features["absolute_return"].dropna()

    if valid_absolute_returns.empty:
        raise ValueError("No valid daily returns were available.")

    thresholds: dict[float, float] = {}

    for percentile in EXTREME_PERCENTILES:
        threshold = float(
            valid_absolute_returns.quantile(percentile)
        )
        thresholds[percentile] = threshold

        percentile_label = int(percentile * 100)
        flag_column = f"is_extreme_p{percentile_label}"

        features[flag_column] = (
            features["absolute_return"] >= threshold
        )

        # The first row has no return and should never be marked extreme.
        features[flag_column] = (
            features[flag_column]
            .fillna(False)
            .astype(bool)
        )

        count = int(features[flag_column].sum())

        print(
            f"P{percentile_label} threshold: "
            f"{threshold * 100:.3f}% | "
            f"extreme days: {count:,}"
        )

    # Main project definition: top 5% of absolute price changes.
    features["is_extreme_day"] = features["is_extreme_p95"]

    return features, thresholds


# ---------------------------------------------------------
# Summary tables
# ---------------------------------------------------------

def create_extreme_days_table(features: pd.DataFrame) -> pd.DataFrame:
    """Creates a table containing only P95 extreme days."""
    useful_columns = [
        "date",
        "brent_price",
        "price_change",
        "daily_return",
        "daily_return_pct",
        "absolute_return",
        "absolute_return_pct",
        "return_direction",
        "is_extreme_p90",
        "is_extreme_p95",
        "is_extreme_p99",
    ]

    # Preserve news variables when they are present in the input file.
    optional_news_columns = [
        "total_headlines",
        "oil_headlines",
        "market_price_headlines",
        "supply_geopolitics_headlines",
        "consumer_fuel_headlines",
        "oil_share",
        "market_price_share",
        "supply_geopolitics_share",
        "consumer_fuel_share",
    ]

    useful_columns.extend(
        column
        for column in optional_news_columns
        if column in features.columns
    )

    extreme_days = features.loc[
        features["is_extreme_day"],
        useful_columns,
    ].copy()

    extreme_days = (
        extreme_days.sort_values(
            "absolute_return",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return extreme_days


def create_yearly_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Summarizes extreme days and volatility by calendar year."""
    summary_data = features.copy()
    summary_data["year"] = summary_data["date"].dt.year

    yearly_summary = (
        summary_data.groupby("year", as_index=False)
        .agg(
            trading_days=("date", "size"),
            mean_brent_price=("brent_price", "mean"),
            mean_daily_return_pct=("daily_return_pct", "mean"),
            mean_absolute_return_pct=("absolute_return_pct", "mean"),
            maximum_absolute_return_pct=("absolute_return_pct", "max"),
            extreme_days_p90=("is_extreme_p90", "sum"),
            extreme_days_p95=("is_extreme_p95", "sum"),
            extreme_days_p99=("is_extreme_p99", "sum"),
        )
    )

    yearly_summary["extreme_day_share_p95"] = (
        yearly_summary["extreme_days_p95"]
        / yearly_summary["trading_days"]
    )

    yearly_summary = yearly_summary.sort_values("year")

    return yearly_summary


# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------

def save_figure(file_path: Path) -> None:
    """Applies common layout settings and saves the active figure."""
    plt.tight_layout()
    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_brent_price(features: pd.DataFrame) -> None:
    """Plots Brent price over the full research period."""
    plt.figure(figsize=(13, 6))
    plt.plot(
        features["date"],
        features["brent_price"],
        linewidth=1,
    )
    plt.title("Daily Brent Crude Oil Price")
    plt.xlabel("Date")
    plt.ylabel("Price (US dollars per barrel)")
    plt.grid(alpha=0.3)
    save_figure(PRICE_PLOT_OUTPUT)


def plot_daily_returns(
    features: pd.DataFrame,
    thresholds: dict[float, float],
) -> None:
    """Plots daily percentage returns with the P95 thresholds."""
    threshold_pct = thresholds[0.95] * 100

    plt.figure(figsize=(13, 6))
    plt.plot(
        features["date"],
        features["daily_return_pct"],
        linewidth=0.7,
    )
    plt.axhline(
        threshold_pct,
        linestyle="--",
        linewidth=1,
        label=f"P95 threshold (+{threshold_pct:.2f}%)",
    )
    plt.axhline(
        -threshold_pct,
        linestyle="--",
        linewidth=1,
        label=f"P95 threshold (-{threshold_pct:.2f}%)",
    )
    plt.axhline(0, linewidth=0.8)
    plt.title("Daily Brent Price Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily return (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(RETURNS_PLOT_OUTPUT)


def plot_rolling_volatility(features: pd.DataFrame) -> None:
    """Plots the rolling 20-trading-day volatility."""
    volatility_column = f"rolling_volatility_{ROLLING_WINDOW}d_pct"

    plt.figure(figsize=(13, 6))
    plt.plot(
        features["date"],
        features[volatility_column],
        linewidth=1,
    )
    plt.title(
        f"Brent Rolling Volatility ({ROLLING_WINDOW} Trading Days)"
    )
    plt.xlabel("Date")
    plt.ylabel("Standard deviation of daily returns (%)")
    plt.grid(alpha=0.3)
    save_figure(VOLATILITY_PLOT_OUTPUT)


def plot_extreme_days(features: pd.DataFrame) -> None:
    """Highlights P95 extreme days on the Brent price series."""
    extreme_days = features[features["is_extreme_day"]]

    plt.figure(figsize=(13, 6))
    plt.plot(
        features["date"],
        features["brent_price"],
        linewidth=1,
        label="Brent price",
    )
    plt.scatter(
        extreme_days["date"],
        extreme_days["brent_price"],
        s=18,
        label="P95 extreme-return day",
        zorder=3,
    )
    plt.title("Brent Price with Extreme Daily Movements")
    plt.xlabel("Date")
    plt.ylabel("Price (US dollars per barrel)")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(EXTREME_PLOT_OUTPUT)


def create_plots(
    features: pd.DataFrame,
    thresholds: dict[float, float],
) -> None:
    """Creates all market-module visualizations."""
    print("\nCreating plots...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_brent_price(features)
    plot_daily_returns(features, thresholds)
    plot_rolling_volatility(features)
    plot_extreme_days(features)

    print(f"Saved: {PRICE_PLOT_OUTPUT}")
    print(f"Saved: {RETURNS_PLOT_OUTPUT}")
    print(f"Saved: {VOLATILITY_PLOT_OUTPUT}")
    print(f"Saved: {EXTREME_PLOT_OUTPUT}")


# ---------------------------------------------------------
# Reporting and saving
# ---------------------------------------------------------

def print_summary(
    features: pd.DataFrame,
    extreme_days: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    thresholds: dict[float, float],
) -> None:
    """Prints key reality-check results to the PyCharm console."""
    valid_returns = features.dropna(subset=["daily_return"])

    largest_increase = valid_returns.loc[
        valid_returns["daily_return"].idxmax()
    ]

    largest_decrease = valid_returns.loc[
        valid_returns["daily_return"].idxmin()
    ]

    highest_volatility_row = features.loc[
        features[f"rolling_volatility_{ROLLING_WINDOW}d"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BRENT MARKET SUMMARY")
    print("=" * 70)

    print(f"Trading dates: {len(features):,}")
    print(
        "Mean daily return: "
        f"{valid_returns['daily_return_pct'].mean():.4f}%"
    )
    print(
        "Mean absolute daily return: "
        f"{valid_returns['absolute_return_pct'].mean():.4f}%"
    )
    print(
        "Median absolute daily return: "
        f"{valid_returns['absolute_return_pct'].median():.4f}%"
    )

    print(
        "Largest daily increase: "
        f"{largest_increase['daily_return_pct']:.2f}% "
        f"on {largest_increase['date'].date()}"
    )

    print(
        "Largest daily decrease: "
        f"{largest_decrease['daily_return_pct']:.2f}% "
        f"on {largest_decrease['date'].date()}"
    )

    print(
        f"Highest {ROLLING_WINDOW}-day rolling volatility: "
        f"{highest_volatility_row[f'rolling_volatility_{ROLLING_WINDOW}d_pct']:.2f}% "
        f"on {highest_volatility_row['date'].date()}"
    )

    print("\nExtreme-day thresholds:")
    for percentile, threshold in thresholds.items():
        percentile_label = int(percentile * 100)
        print(
            f"P{percentile_label}: "
            f"absolute daily return >= {threshold * 100:.3f}%"
        )

    print(
        "\nNumber of main P95 extreme days: "
        f"{len(extreme_days):,}"
    )

    print("\nTen largest absolute daily movements:")
    print(
        extreme_days[
            [
                "date",
                "brent_price",
                "daily_return_pct",
                "return_direction",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            formatters={
                "brent_price": lambda value: f"{value:.2f}",
                "daily_return_pct": lambda value: f"{value:.2f}%",
            },
        )
    )

    print("\nTen years with the most P95 extreme days:")
    print(
        yearly_summary.sort_values(
            ["extreme_days_p95", "year"],
            ascending=[False, True],
        )
        .head(10)[
            [
                "year",
                "trading_days",
                "extreme_days_p95",
                "extreme_day_share_p95",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "extreme_day_share_p95": (
                    lambda value: f"{value:.2%}"
                )
            },
        )
    )


def save_results(
    features: pd.DataFrame,
    extreme_days: pd.DataFrame,
    yearly_summary: pd.DataFrame,
) -> None:
    """Saves all market-module data tables."""
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        FEATURES_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    extreme_days.to_csv(
        EXTREME_DAYS_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )

    yearly_summary.to_csv(
        YEARLY_SUMMARY_OUTPUT,
        index=False,
    )

    print("\nSaved processed files:")
    print(f"1. {FEATURES_OUTPUT}")
    print(f"2. {EXTREME_DAYS_OUTPUT}")
    print(f"3. {YEARLY_SUMMARY_OUTPUT}")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("GeoOil-Pulse: Brent Market and Volatility Analysis")
    print("=" * 70)

    check_file_exists(INPUT_FILE)

    data = load_data(INPUT_FILE)
    features = add_return_features(data)

    features, thresholds = add_extreme_day_flags(features)

    extreme_days = create_extreme_days_table(features)
    yearly_summary = create_yearly_summary(features)

    save_results(
        features=features,
        extreme_days=extreme_days,
        yearly_summary=yearly_summary,
    )

    create_plots(
        features=features,
        thresholds=thresholds,
    )

    print_summary(
        features=features,
        extreme_days=extreme_days,
        yearly_summary=yearly_summary,
        thresholds=thresholds,
    )

    print("\nBrent volatility analysis completed successfully.")


if __name__ == "__main__":
    main()
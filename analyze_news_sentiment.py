from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import kruskal, mannwhitneyu


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

HEADLINES_INPUT_FILE = (
    PROCESSED_DATA_DIR / "oil_headlines_categorized.csv"
)
MARKET_INPUT_FILE = (
    PROCESSED_DATA_DIR / "brent_market_features.csv"
)

HEADLINES_WITH_SENTIMENT_OUTPUT = (
    PROCESSED_DATA_DIR / "oil_headlines_with_sentiment.csv"
)
DAILY_SENTIMENT_OUTPUT = (
    PROCESSED_DATA_DIR / "daily_news_sentiment.csv"
)
MERGED_SENTIMENT_OUTPUT = (
    PROCESSED_DATA_DIR / "brent_news_sentiment_daily.csv"
)
VALIDATION_SAMPLE_OUTPUT = (
    OUTPUT_DIR / "sample_sentiment_validation.csv"
)
LABEL_SUMMARY_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_label_summary.csv"
)
EXTREME_COMPARISON_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_extreme_vs_normal_comparison.csv"
)
DIRECTION_SUMMARY_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_direction_summary.csv"
)
DIRECTION_TESTS_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_direction_tests.csv"
)
EVENTS_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_market_events_p95.csv"
)
EVENT_STUDY_LONG_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_event_study_long.csv"
)
EVENT_STUDY_SUMMARY_OUTPUT = (
    PROCESSED_DATA_DIR / "sentiment_event_study_summary.csv"
)

EXTREME_COMPARISON_PLOT = (
    OUTPUT_DIR / "sentiment_extreme_vs_normal.png"
)
DIRECTION_PLOT = (
    OUTPUT_DIR / "sentiment_by_market_direction.png"
)
EVENT_STUDY_PLOT = (
    OUTPUT_DIR / "sentiment_event_study.png"
)


# ---------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------

# Standard VADER thresholds.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

MAIN_EXTREME_FLAG = "is_extreme_p95"
EVENT_WINDOW = 5
MAX_GAP_WITHIN_EPISODE = 2 * EVENT_WINDOW
VALIDATION_SAMPLE_PER_LABEL = 50
RANDOM_SEED = 42

CATEGORY_DEFINITIONS = {
    "all_oil": {
        "filter_column": None,
        "label": "All oil coverage",
        "prefix": "",
    },
    "market_prices": {
        "filter_column": "is_market_prices",
        "label": "Market and prices",
        "prefix": "market_",
    },
    "supply_geopolitics": {
        "filter_column": "is_supply_geopolitics",
        "label": "Supply and geopolitics",
        "prefix": "supply_",
    },
    "consumer_fuel": {
        "filter_column": "is_consumer_fuel",
        "label": "Consumer fuel",
        "prefix": "consumer_",
    },
}

# Daily sentiment measures used in the statistical comparisons.
SENTIMENT_MEASURES = {
    "mean_sentiment": "Mean VADER compound score",
    "negative_headline_share": "Negative-headline share",
    "positive_headline_share": "Positive-headline share",
}


# ---------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------


def check_file_exists(file_path: Path, previous_step: str) -> None:
    """Verifies that a required input file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the file:\n{file_path}\n\n"
            f"Run {previous_step} first."
        )


def parse_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    """Converts True/False or 0/1 values to a Boolean series."""
    if series.dtype == bool:
        return series

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    converted = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    if converted.isna().any():
        invalid_values = sorted(
            normalized[converted.isna()].dropna().unique().tolist()
        )
        raise ValueError(
            f"Column {column_name} contains invalid Boolean values: "
            f"{invalid_values[:10]}"
        )

    return converted.astype(bool)


def load_headlines(file_path: Path) -> pd.DataFrame:
    """Loads the categorized oil headlines."""
    print("\nLoading categorized oil headlines...")

    headlines = pd.read_csv(
        file_path,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "headline_text",
        "primary_category",
        "categories",
        "is_market_prices",
        "is_supply_geopolitics",
        "is_consumer_fuel",
    }

    missing_columns = required_columns - set(headlines.columns)

    if missing_columns:
        raise ValueError(
            "The headlines file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    headlines["headline_text"] = (
        headlines["headline_text"]
        .astype("string")
        .str.strip()
    )

    invalid_rows = (
        headlines["date"].isna()
        | headlines["headline_text"].isna()
        | (headlines["headline_text"] == "")
    )

    invalid_count = int(invalid_rows.sum())

    if invalid_count:
        print(f"Invalid headline rows removed: {invalid_count:,}")
        headlines = headlines.loc[~invalid_rows].copy()

    rows_before_duplicates = len(headlines)
    headlines = headlines.drop_duplicates(
        subset=["date", "headline_text"],
        keep="first",
    )
    duplicate_count = rows_before_duplicates - len(headlines)

    if duplicate_count:
        print(f"Duplicate date-headline rows removed: {duplicate_count:,}")

    boolean_columns = [
        "is_market_prices",
        "is_supply_geopolitics",
        "is_consumer_fuel",
    ]

    for column in boolean_columns:
        headlines[column] = parse_boolean_series(
            headlines[column],
            column,
        )

    headlines = (
        headlines.sort_values(["date", "headline_text"])
        .reset_index(drop=True)
    )

    print(f"Headlines loaded: {len(headlines):,}")
    print(
        "Headline date range: "
        f"{headlines['date'].min().date()} "
        f"to {headlines['date'].max().date()}"
    )

    return headlines


def load_market_features(file_path: Path) -> pd.DataFrame:
    """Loads Brent market features used for the comparison analyses."""
    print("\nLoading Brent market features...")

    market = pd.read_csv(
        file_path,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "brent_price",
        "daily_return",
        "daily_return_pct",
        "absolute_return",
        "absolute_return_pct",
        "return_direction",
        "oil_headlines",
        MAIN_EXTREME_FLAG,
    }

    missing_columns = required_columns - set(market.columns)

    if missing_columns:
        raise ValueError(
            "The market-features file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    market = (
        market.sort_values("date")
        .reset_index(drop=True)
    )

    if market["date"].duplicated().any():
        raise ValueError("The market-features file contains duplicate dates.")

    numeric_columns = [
        "brent_price",
        "daily_return",
        "daily_return_pct",
        "absolute_return",
        "absolute_return_pct",
        "oil_headlines",
    ]

    for column in numeric_columns:
        market[column] = pd.to_numeric(
            market[column],
            errors="coerce",
        )

    market[MAIN_EXTREME_FLAG] = parse_boolean_series(
        market[MAIN_EXTREME_FLAG],
        MAIN_EXTREME_FLAG,
    )

    print(f"Market rows loaded: {len(market):,}")
    print(
        "Market date range: "
        f"{market['date'].min().date()} "
        f"to {market['date'].max().date()}"
    )

    return market


# ---------------------------------------------------------
# VADER sentiment scoring
# ---------------------------------------------------------


def initialize_vader() -> SentimentIntensityAnalyzer:
    """
    Initializes VADER.

    NLTK downloads the small vader_lexicon resource once if it is not
    already installed on the computer.
    """
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        print("\nVADER lexicon was not found. Downloading it once...")

        download_succeeded = nltk.download(
            "vader_lexicon",
            quiet=True,
        )

        if not download_succeeded:
            raise RuntimeError(
                "Could not download the NLTK VADER lexicon.\n"
                "Run this command in the PyCharm terminal and retry:\n"
                "python -m nltk.downloader vader_lexicon"
            )

        try:
            return SentimentIntensityAnalyzer()
        except LookupError as error:
            raise RuntimeError(
                "The VADER lexicon is still unavailable.\n"
                "Run this command in the PyCharm terminal:\n"
                "python -m nltk.downloader vader_lexicon"
            ) from error


def sentiment_label(compound_score: float) -> str:
    """Maps a VADER compound score to positive, neutral or negative."""
    if compound_score >= POSITIVE_THRESHOLD:
        return "positive"

    if compound_score <= NEGATIVE_THRESHOLD:
        return "negative"

    return "neutral"


def score_headlines(
    headlines: pd.DataFrame,
    analyzer: SentimentIntensityAnalyzer,
) -> pd.DataFrame:
    """Adds VADER sentiment scores and labels to each headline."""
    print("\nScoring headline sentiment with VADER...")

    score_records = [
        analyzer.polarity_scores(str(text))
        for text in headlines["headline_text"]
    ]

    scores = pd.DataFrame(score_records)

    required_score_columns = {"compound", "pos", "neg", "neu"}

    if not required_score_columns.issubset(scores.columns):
        raise ValueError(
            "VADER did not return the expected sentiment score fields."
        )

    scored = headlines.copy()

    scored["sentiment_compound"] = scores["compound"].to_numpy()
    scored["sentiment_positive"] = scores["pos"].to_numpy()
    scored["sentiment_negative"] = scores["neg"].to_numpy()
    scored["sentiment_neutral"] = scores["neu"].to_numpy()

    scored["sentiment_label"] = scored[
        "sentiment_compound"
    ].map(sentiment_label)

    print("Sentiment scoring completed.")
    print("\nSentiment-label counts:")
    print(
        scored["sentiment_label"]
        .value_counts()
        .reindex(["negative", "neutral", "positive"], fill_value=0)
        .to_string()
    )

    return scored


# ---------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------


def summarize_daily_subset(
    subset: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Creates daily sentiment measures for one headline category."""
    if subset.empty:
        return pd.DataFrame(columns=["date"])

    daily = (
        subset.groupby("date")
        .agg(
            headline_count=("sentiment_compound", "size"),
            mean_sentiment=("sentiment_compound", "mean"),
            median_sentiment=("sentiment_compound", "median"),
            mean_positive_score=("sentiment_positive", "mean"),
            mean_negative_score=("sentiment_negative", "mean"),
            mean_neutral_score=("sentiment_neutral", "mean"),
            positive_headlines=(
                "sentiment_label",
                lambda values: int((values == "positive").sum()),
            ),
            negative_headlines=(
                "sentiment_label",
                lambda values: int((values == "negative").sum()),
            ),
            neutral_headlines=(
                "sentiment_label",
                lambda values: int((values == "neutral").sum()),
            ),
        )
        .reset_index()
    )

    daily["positive_headline_share"] = (
        daily["positive_headlines"]
        / daily["headline_count"]
    )
    daily["negative_headline_share"] = (
        daily["negative_headlines"]
        / daily["headline_count"]
    )
    daily["neutral_headline_share"] = (
        daily["neutral_headlines"]
        / daily["headline_count"]
    )

    if prefix == "":
        rename_map = {
            "headline_count": "sentiment_headline_count",
        }
    else:
        rename_map = {
            column: f"{prefix}{column}"
            for column in daily.columns
            if column != "date"
        }

    daily = daily.rename(columns=rename_map)

    return daily


def create_daily_sentiment_table(
    scored_headlines: pd.DataFrame,
) -> pd.DataFrame:
    """Creates daily sentiment measures for all four analysis groups."""
    print("\nCreating daily sentiment measures...")

    daily_tables: list[pd.DataFrame] = []

    for category_info in CATEGORY_DEFINITIONS.values():
        filter_column = category_info["filter_column"]
        prefix = category_info["prefix"]

        if filter_column is None:
            subset = scored_headlines
        else:
            subset = scored_headlines.loc[
                scored_headlines[filter_column]
            ]

        category_daily = summarize_daily_subset(
            subset=subset,
            prefix=prefix,
        )

        daily_tables.append(category_daily)

    daily = daily_tables[0]

    for category_daily in daily_tables[1:]:
        daily = daily.merge(
            category_daily,
            on="date",
            how="outer",
            validate="one_to_one",
        )

    count_columns = [
        column
        for column in daily.columns
        if (
            column.endswith("headline_count")
            or column.endswith("positive_headlines")
            or column.endswith("negative_headlines")
            or column.endswith("neutral_headlines")
        )
    ]

    for column in count_columns:
        daily[column] = (
            daily[column]
            .fillna(0)
            .astype(int)
        )

    daily = (
        daily.sort_values("date")
        .reset_index(drop=True)
    )

    print(f"Dates with at least one oil headline: {len(daily):,}")

    return daily


def merge_sentiment_with_market(
    market: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges daily sentiment with Brent market features.

    Sentiment values remain missing on trading days with no relevant
    headline. They are not replaced by zero because zero would falsely
    mean neutral sentiment rather than no coverage.
    """
    print("\nMerging daily sentiment with Brent market features...")

    merged = market.merge(
        daily_sentiment,
        on="date",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        column
        for column in daily_sentiment.columns
        if (
            column.endswith("headline_count")
            or column.endswith("positive_headlines")
            or column.endswith("negative_headlines")
            or column.endswith("neutral_headlines")
        )
    ]

    for column in count_columns:
        merged[column] = (
            merged[column]
            .fillna(0)
            .astype(int)
        )

    # Reality check: the all-oil sentiment count should match the oil
    # headline count already stored in brent_market_features.csv.
    mismatches = (
        merged["sentiment_headline_count"]
        != merged["oil_headlines"].fillna(0).astype(int)
    )

    mismatch_count = int(mismatches.sum())

    if mismatch_count:
        raise ValueError(
            "Headline-count mismatch detected on "
            f"{mismatch_count:,} trading dates. "
            "Run filter_oil_headlines.py and "
            "analyze_brent_volatility.py again before this script."
        )

    merged["has_oil_sentiment"] = (
        merged["sentiment_headline_count"] > 0
    )

    no_coverage_days = int((~merged["has_oil_sentiment"]).sum())

    print(f"Merged trading dates: {len(merged):,}")
    print(
        "Trading dates without an oil headline: "
        f"{no_coverage_days:,}"
    )

    return merged


# ---------------------------------------------------------
# Validation sample and label summary
# ---------------------------------------------------------


def create_validation_sample(
    scored_headlines: pd.DataFrame,
) -> pd.DataFrame:
    """Creates a balanced sample for manual sentiment validation."""
    samples: list[pd.DataFrame] = []

    for label in ["negative", "neutral", "positive"]:
        label_rows = scored_headlines.loc[
            scored_headlines["sentiment_label"] == label
        ]

        sample_size = min(
            VALIDATION_SAMPLE_PER_LABEL,
            len(label_rows),
        )

        if sample_size == 0:
            continue

        sample = label_rows.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )

        samples.append(sample)

    if samples:
        validation = pd.concat(
            samples,
            ignore_index=True,
        )
    else:
        validation = scored_headlines.head(0).copy()

    validation = validation[
        [
            "date",
            "headline_text",
            "primary_category",
            "categories",
            "sentiment_compound",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_label",
        ]
    ].sort_values(
        ["sentiment_label", "date"]
    )

    # Empty fields to be completed manually in Excel or PyCharm.
    validation["manual_label"] = ""
    validation["is_model_correct"] = ""
    validation["review_notes"] = ""

    return validation.reset_index(drop=True)


def create_label_summary(
    scored_headlines: pd.DataFrame,
) -> pd.DataFrame:
    """Summarizes VADER labels overall and by primary category."""
    overall = (
        scored_headlines["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment_label")
        .reset_index(name="headline_count")
    )
    overall["group"] = "all_oil"

    category_summary = (
        scored_headlines.groupby(
            ["primary_category", "sentiment_label"]
        )
        .size()
        .reset_index(name="headline_count")
        .rename(columns={"primary_category": "group"})
    )

    summary = pd.concat(
        [overall, category_summary],
        ignore_index=True,
    )

    group_totals = (
        summary.groupby("group")["headline_count"]
        .transform("sum")
    )
    summary["share_within_group"] = (
        summary["headline_count"] / group_totals
    )

    return summary.sort_values(
        ["group", "sentiment_label"]
    ).reset_index(drop=True)


# ---------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Applies Benjamini-Hochberg FDR correction."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)

    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        return pd.Series(adjusted, index=p_values.index)

    order = np.argsort(valid_values)
    ranked = valid_values[order]
    ranks = np.arange(1, len(ranked) + 1)

    ranked_adjusted = ranked * len(ranked) / ranks
    ranked_adjusted = np.minimum.accumulate(
        ranked_adjusted[::-1]
    )[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0, 1)

    valid_adjusted = np.empty_like(ranked_adjusted)
    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_mask] = valid_adjusted

    return pd.Series(adjusted, index=p_values.index)


def daily_metric_columns(prefix: str) -> dict[str, str]:
    """Returns the daily sentiment columns for one category prefix."""
    if prefix == "":
        return {
            "mean_sentiment": "mean_sentiment",
            "negative_headline_share": "negative_headline_share",
            "positive_headline_share": "positive_headline_share",
            "count": "sentiment_headline_count",
        }

    return {
        "mean_sentiment": f"{prefix}mean_sentiment",
        "negative_headline_share": f"{prefix}negative_headline_share",
        "positive_headline_share": f"{prefix}positive_headline_share",
        "count": f"{prefix}headline_count",
    }


# ---------------------------------------------------------
# Extreme versus normal days
# ---------------------------------------------------------


def compare_extreme_and_normal_days(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compares daily sentiment on P95 extreme and normal days."""
    print("\nComparing sentiment on extreme and normal trading days...")

    rows: list[dict[str, float | int | str | bool]] = []

    for category_name, category_info in CATEGORY_DEFINITIONS.items():
        columns = daily_metric_columns(category_info["prefix"])

        for measure_name, measure_label in SENTIMENT_MEASURES.items():
            value_column = columns[measure_name]

            valid = data.loc[
                data["daily_return"].notna()
                & data[value_column].notna(),
                [value_column, MAIN_EXTREME_FLAG],
            ]

            normal = valid.loc[
                ~valid[MAIN_EXTREME_FLAG],
                value_column,
            ]
            extreme = valid.loc[
                valid[MAIN_EXTREME_FLAG],
                value_column,
            ]

            if len(normal) >= 2 and len(extreme) >= 2:
                test = mannwhitneyu(
                    extreme,
                    normal,
                    alternative="two-sided",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                statistic = np.nan
                p_value = np.nan

            rows.append(
                {
                    "category": category_name,
                    "category_label": category_info["label"],
                    "measure": measure_name,
                    "measure_label": measure_label,
                    "normal_n_days_with_coverage": len(normal),
                    "extreme_n_days_with_coverage": len(extreme),
                    "normal_mean": float(normal.mean()),
                    "extreme_mean": float(extreme.mean()),
                    "mean_difference_extreme_minus_normal": float(
                        extreme.mean() - normal.mean()
                    ),
                    "normal_median": float(normal.median()),
                    "extreme_median": float(extreme.median()),
                    "mann_whitney_u": statistic,
                    "p_value": p_value,
                }
            )

    comparison = pd.DataFrame(rows)
    comparison["fdr_p_value"] = benjamini_hochberg(
        comparison["p_value"]
    )
    comparison["significant_fdr_0_05"] = (
        comparison["fdr_p_value"] < 0.05
    )

    return comparison


# ---------------------------------------------------------
# Normal, extreme increase and extreme decrease
# ---------------------------------------------------------


def add_market_direction_group(data: pd.DataFrame) -> pd.DataFrame:
    """Adds normal, extreme-increase and extreme-decrease labels."""
    grouped = data.copy()

    grouped["sentiment_market_group"] = np.select(
        [
            grouped[MAIN_EXTREME_FLAG]
            & (grouped["daily_return"] > 0),
            grouped[MAIN_EXTREME_FLAG]
            & (grouped["daily_return"] < 0),
        ],
        [
            "extreme_increase",
            "extreme_decrease",
        ],
        default="normal",
    )

    return grouped


def summarize_direction_groups(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarizes and tests sentiment across the three market groups."""
    print("\nComparing sentiment by direction of market movement...")

    summary_rows: list[dict[str, float | int | str]] = []
    test_rows: list[dict[str, float | int | str | bool]] = []

    group_order = [
        "normal",
        "extreme_increase",
        "extreme_decrease",
    ]

    for category_name, category_info in CATEGORY_DEFINITIONS.items():
        columns = daily_metric_columns(category_info["prefix"])

        for measure_name, measure_label in SENTIMENT_MEASURES.items():
            value_column = columns[measure_name]
            group_values: list[pd.Series] = []

            for group_name in group_order:
                values = data.loc[
                    (data["sentiment_market_group"] == group_name)
                    & data[value_column].notna()
                    & data["daily_return"].notna(),
                    value_column,
                ]

                group_values.append(values)

                summary_rows.append(
                    {
                        "category": category_name,
                        "category_label": category_info["label"],
                        "measure": measure_name,
                        "measure_label": measure_label,
                        "market_group": group_name,
                        "n_days_with_coverage": len(values),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "standard_deviation": float(values.std(ddof=1)),
                    }
                )

            if all(len(values) >= 2 for values in group_values):
                test = kruskal(*group_values)
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                statistic = np.nan
                p_value = np.nan

            test_rows.append(
                {
                    "category": category_name,
                    "category_label": category_info["label"],
                    "measure": measure_name,
                    "measure_label": measure_label,
                    "kruskal_wallis_h": statistic,
                    "p_value": p_value,
                }
            )

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)

    tests["fdr_p_value"] = benjamini_hochberg(
        tests["p_value"]
    )
    tests["significant_fdr_0_05"] = (
        tests["fdr_p_value"] < 0.05
    )

    return summary, tests


# ---------------------------------------------------------
# Independent P95 episodes and sentiment event study
# ---------------------------------------------------------


def identify_independent_extreme_episodes(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Groups nearby P95 days and chooses the strongest day as anchor."""
    print("\nIdentifying independent P95 market episodes...")

    valid = (
        data.loc[data["daily_return"].notna()]
        .reset_index(drop=True)
        .copy()
    )

    extreme_positions = np.flatnonzero(
        valid[MAIN_EXTREME_FLAG].to_numpy(dtype=bool)
    )

    if len(extreme_positions) == 0:
        raise ValueError("No P95 extreme days were found.")

    episode_groups: list[list[int]] = []
    current_group = [int(extreme_positions[0])]

    for raw_position in extreme_positions[1:]:
        position = int(raw_position)

        if position - current_group[-1] <= MAX_GAP_WITHIN_EPISODE:
            current_group.append(position)
        else:
            episode_groups.append(current_group)
            current_group = [position]

    episode_groups.append(current_group)

    rows: list[dict[str, float | int | str | bool | pd.Timestamp]] = []

    for event_id, positions in enumerate(episode_groups, start=1):
        episode_extremes = valid.iloc[positions]
        anchor_position = int(
            episode_extremes["absolute_return"].idxmax()
        )
        anchor = valid.iloc[anchor_position]

        complete_window = (
            anchor_position - EVENT_WINDOW >= 0
            and anchor_position + EVENT_WINDOW < len(valid)
        )

        rows.append(
            {
                "event_id": event_id,
                "episode_first_extreme_date": valid.iloc[
                    positions[0]
                ]["date"],
                "episode_last_extreme_date": valid.iloc[
                    positions[-1]
                ]["date"],
                "n_p95_extreme_days_in_episode": len(positions),
                "anchor_date": anchor["date"],
                "anchor_position": anchor_position,
                "anchor_brent_price": float(anchor["brent_price"]),
                "anchor_daily_return_pct": float(
                    anchor["daily_return_pct"]
                ),
                "anchor_absolute_return_pct": float(
                    anchor["absolute_return_pct"]
                ),
                "anchor_direction": anchor["return_direction"],
                "event_window_complete": bool(complete_window),
            }
        )

    events = pd.DataFrame(rows)

    print(f"Independent episodes identified: {len(events):,}")
    print(
        f"Episodes with complete ±{EVENT_WINDOW} window: "
        f"{int(events['event_window_complete'].sum()):,}"
    )

    return events


def build_sentiment_event_study(
    data: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Builds one row per event and relative trading day."""
    valid = (
        data.loc[data["daily_return"].notna()]
        .reset_index(drop=True)
        .copy()
    )

    complete_events = events.loc[
        events["event_window_complete"]
    ]

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for _, event in complete_events.iterrows():
        anchor_position = int(event["anchor_position"])

        for relative_day in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
            observation = valid.iloc[
                anchor_position + relative_day
            ]

            row: dict[str, float | int | str | pd.Timestamp] = {
                "event_id": int(event["event_id"]),
                "anchor_date": event["anchor_date"],
                "anchor_direction": event["anchor_direction"],
                "anchor_absolute_return_pct": float(
                    event["anchor_absolute_return_pct"]
                ),
                "relative_trading_day": relative_day,
                "observation_date": observation["date"],
            }

            for category_name, category_info in CATEGORY_DEFINITIONS.items():
                columns = daily_metric_columns(category_info["prefix"])

                row[f"{category_name}_mean_sentiment"] = observation[
                    columns["mean_sentiment"]
                ]
                row[f"{category_name}_negative_share"] = observation[
                    columns["negative_headline_share"]
                ]
                row[f"{category_name}_headline_count"] = int(
                    observation[columns["count"]]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_sentiment_event_study(
    event_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarizes mean sentiment around event anchors.

    The pre-event baseline is the pooled average over trading days -5
    through -2. Day -1 is omitted to reduce contamination from news that
    may already be reacting to the developing event.
    """
    rows: list[dict[str, float | int | str]] = []

    for category_name, category_info in CATEGORY_DEFINITIONS.items():
        value_column = f"{category_name}_mean_sentiment"

        baseline_values = event_long.loc[
            event_long["relative_trading_day"].between(
                -EVENT_WINDOW,
                -2,
            ),
            value_column,
        ].dropna()

        baseline_mean = float(baseline_values.mean())

        for relative_day, group in event_long.groupby(
            "relative_trading_day"
        ):
            values = group[value_column].dropna()

            rows.append(
                {
                    "category": category_name,
                    "category_label": category_info["label"],
                    "relative_trading_day": int(relative_day),
                    "n_events_with_sentiment": len(values),
                    "mean_sentiment": float(values.mean()),
                    "median_sentiment": float(values.median()),
                    "pre_event_baseline_mean": baseline_mean,
                    "mean_excess_vs_pre_event_baseline": float(
                        values.mean() - baseline_mean
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------


def plot_extreme_vs_normal(
    comparison: pd.DataFrame,
) -> None:
    """Plots mean daily VADER sentiment on normal and P95 days."""
    plot_data = comparison.loc[
        comparison["measure"] == "mean_sentiment"
    ].copy()

    labels = plot_data["category_label"].tolist()
    normal_means = plot_data["normal_mean"].to_numpy()
    extreme_means = plot_data["extreme_mean"].to_numpy()

    positions = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        positions - width / 2,
        normal_means,
        width,
        label="Normal trading days",
    )
    ax.bar(
        positions + width / 2,
        extreme_means,
        width,
        label="P95 extreme-return days",
    )

    ax.axhline(0, linewidth=1)
    ax.set_title(
        "Mean Oil-News Sentiment on Extreme and Normal Brent Trading Days"
    )
    ax.set_xlabel("News category")
    ax.set_ylabel("Mean VADER compound score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        EXTREME_COMPARISON_PLOT,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_direction_groups(
    direction_summary: pd.DataFrame,
) -> None:
    """Plots mean daily sentiment by direction of market movement."""
    plot_data = direction_summary.loc[
        direction_summary["measure"] == "mean_sentiment"
    ].copy()

    categories = [
        info["label"]
        for info in CATEGORY_DEFINITIONS.values()
    ]
    group_order = [
        "normal",
        "extreme_increase",
        "extreme_decrease",
    ]
    group_labels = {
        "normal": "Normal",
        "extreme_increase": "Extreme increase",
        "extreme_decrease": "Extreme decrease",
    }

    positions = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    for index, group_name in enumerate(group_order):
        group_data = plot_data.loc[
            plot_data["market_group"] == group_name
        ].set_index("category_label")

        values = [
            group_data.loc[label, "mean"]
            if label in group_data.index
            else np.nan
            for label in categories
        ]

        offset = (index - 1) * width

        ax.bar(
            positions + offset,
            values,
            width,
            label=group_labels[group_name],
        )

    ax.axhline(0, linewidth=1)
    ax.set_title(
        "Oil-News Sentiment by Direction of Extreme Brent Movement"
    )
    ax.set_xlabel("News category")
    ax.set_ylabel("Mean VADER compound score")
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        DIRECTION_PLOT,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_event_study(event_summary: pd.DataFrame) -> None:
    """Plots sentiment relative to the pooled pre-event baseline."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for category_name, category_info in CATEGORY_DEFINITIONS.items():
        category_data = event_summary.loc[
            event_summary["category"] == category_name
        ].sort_values("relative_trading_day")

        ax.plot(
            category_data["relative_trading_day"],
            category_data["mean_excess_vs_pre_event_baseline"],
            marker="o",
            label=category_info["label"],
        )

    ax.axvline(
        0,
        linestyle="--",
        label="Extreme-market event",
    )
    ax.axhline(0, linewidth=1)
    ax.set_title(
        "Oil-News Sentiment Around Independent P95 Market Episodes"
    )
    ax.set_xlabel("Trading days relative to event anchor")
    ax.set_ylabel(
        "Mean VADER sentiment minus pooled pre-event baseline"
    )
    ax.set_xticks(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        EVENT_STUDY_PLOT,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------
# Saving and printed summary
# ---------------------------------------------------------


def save_results(
    scored_headlines: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    merged: pd.DataFrame,
    validation_sample: pd.DataFrame,
    label_summary: pd.DataFrame,
    extreme_comparison: pd.DataFrame,
    direction_summary: pd.DataFrame,
    direction_tests: pd.DataFrame,
    events: pd.DataFrame,
    event_long: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> None:
    """Saves all processed tables."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scored_headlines.to_csv(
        HEADLINES_WITH_SENTIMENT_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    daily_sentiment.to_csv(
        DAILY_SENTIMENT_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    merged.to_csv(
        MERGED_SENTIMENT_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    validation_sample.to_csv(
        VALIDATION_SAMPLE_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    label_summary.to_csv(
        LABEL_SUMMARY_OUTPUT,
        index=False,
    )
    extreme_comparison.to_csv(
        EXTREME_COMPARISON_OUTPUT,
        index=False,
    )
    direction_summary.to_csv(
        DIRECTION_SUMMARY_OUTPUT,
        index=False,
    )
    direction_tests.to_csv(
        DIRECTION_TESTS_OUTPUT,
        index=False,
    )
    events.to_csv(
        EVENTS_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    event_long.to_csv(
        EVENT_STUDY_LONG_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    event_summary.to_csv(
        EVENT_STUDY_SUMMARY_OUTPUT,
        index=False,
    )

    print("\nSaved data files:")
    for path in [
        HEADLINES_WITH_SENTIMENT_OUTPUT,
        DAILY_SENTIMENT_OUTPUT,
        MERGED_SENTIMENT_OUTPUT,
        LABEL_SUMMARY_OUTPUT,
        EXTREME_COMPARISON_OUTPUT,
        DIRECTION_SUMMARY_OUTPUT,
        DIRECTION_TESTS_OUTPUT,
        EVENTS_OUTPUT,
        EVENT_STUDY_LONG_OUTPUT,
        EVENT_STUDY_SUMMARY_OUTPUT,
        VALIDATION_SAMPLE_OUTPUT,
    ]:
        print(f"- {path}")

    print("\nSaved plots:")
    for path in [
        EXTREME_COMPARISON_PLOT,
        DIRECTION_PLOT,
        EVENT_STUDY_PLOT,
    ]:
        print(f"- {path}")


def print_key_results(
    scored_headlines: pd.DataFrame,
    extreme_comparison: pd.DataFrame,
    direction_tests: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    """Prints a concise reality-check summary."""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"Headlines scored: {len(scored_headlines):,}")

    label_counts = (
        scored_headlines["sentiment_label"]
        .value_counts()
        .reindex(["negative", "neutral", "positive"], fill_value=0)
    )

    for label, count in label_counts.items():
        share = count / len(scored_headlines)
        print(f"{label.title()}: {count:,} ({share:.1%})")

    print(
        "\nImportant interpretation: VADER measures linguistic tone. "
        "A phrase such as 'oil prices rise' may receive a positive "
        "score even though higher prices are not beneficial to every "
        "consumer or business."
    )

    main_results = extreme_comparison.loc[
        extreme_comparison["measure"] == "mean_sentiment",
        [
            "category_label",
            "normal_mean",
            "extreme_mean",
            "mean_difference_extreme_minus_normal",
            "fdr_p_value",
            "significant_fdr_0_05",
        ],
    ]

    print("\nMean sentiment: P95 extreme versus normal days")
    print(
        main_results.to_string(
            index=False,
            formatters={
                "normal_mean": lambda value: f"{value:.4f}",
                "extreme_mean": lambda value: f"{value:.4f}",
                "mean_difference_extreme_minus_normal": (
                    lambda value: f"{value:.4f}"
                ),
                "fdr_p_value": lambda value: f"{value:.6f}",
            },
        )
    )

    main_direction_tests = direction_tests.loc[
        direction_tests["measure"] == "mean_sentiment",
        [
            "category_label",
            "kruskal_wallis_h",
            "fdr_p_value",
            "significant_fdr_0_05",
        ],
    ]

    print("\nKruskal-Wallis tests across normal/increase/decrease groups")
    print(
        main_direction_tests.to_string(
            index=False,
            formatters={
                "kruskal_wallis_h": lambda value: f"{value:.4f}",
                "fdr_p_value": lambda value: f"{value:.6f}",
            },
        )
    )

    print(
        "\nIndependent P95 episodes: "
        f"{len(events):,}"
    )

    print(
        "\nManual validation is required before treating VADER labels "
        "as final. Open sample_sentiment_validation.csv and complete "
        "manual_label, is_model_correct and review_notes."
    )


def main() -> None:
    """Runs the complete news-sentiment analysis pipeline."""
    print("=" * 70)
    print("GeoOil-Pulse: Oil-News Sentiment Analysis")
    print("=" * 70)

    check_file_exists(
        HEADLINES_INPUT_FILE,
        "filter_oil_headlines.py",
    )
    check_file_exists(
        MARKET_INPUT_FILE,
        "analyze_brent_volatility.py",
    )

    headlines = load_headlines(HEADLINES_INPUT_FILE)
    market = load_market_features(MARKET_INPUT_FILE)

    analyzer = initialize_vader()
    scored_headlines = score_headlines(
        headlines=headlines,
        analyzer=analyzer,
    )

    daily_sentiment = create_daily_sentiment_table(
        scored_headlines
    )
    merged = merge_sentiment_with_market(
        market=market,
        daily_sentiment=daily_sentiment,
    )
    merged = add_market_direction_group(merged)

    validation_sample = create_validation_sample(
        scored_headlines
    )
    label_summary = create_label_summary(
        scored_headlines
    )

    extreme_comparison = compare_extreme_and_normal_days(
        merged
    )
    direction_summary, direction_tests = (
        summarize_direction_groups(merged)
    )

    events = identify_independent_extreme_episodes(merged)
    event_long = build_sentiment_event_study(
        data=merged,
        events=events,
    )
    event_summary = summarize_sentiment_event_study(
        event_long
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_extreme_vs_normal(extreme_comparison)
    plot_direction_groups(direction_summary)
    plot_event_study(event_summary)

    save_results(
        scored_headlines=scored_headlines,
        daily_sentiment=daily_sentiment,
        merged=merged,
        validation_sample=validation_sample,
        label_summary=label_summary,
        extreme_comparison=extreme_comparison,
        direction_summary=direction_summary,
        direction_tests=direction_tests,
        events=events,
        event_long=event_long,
        event_summary=event_summary,
    )

    print_key_results(
        scored_headlines=scored_headlines,
        extreme_comparison=extreme_comparison,
        direction_tests=direction_tests,
        events=events,
    )

    print("\nSentiment analysis completed successfully.")


if __name__ == "__main__":
    main()
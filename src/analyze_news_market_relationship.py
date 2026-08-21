from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, spearmanr


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

INPUT_FILE = PROCESSED_DATA_DIR / "brent_market_features.csv"

ANALYSIS_DATA_OUTPUT = (
    PROCESSED_DATA_DIR / "news_market_relationship_data.csv"
)
GROUP_COMPARISON_OUTPUT = (
    PROCESSED_DATA_DIR / "news_extreme_vs_normal_comparison.csv"
)
DIRECTION_SUMMARY_OUTPUT = (
    PROCESSED_DATA_DIR / "news_extreme_direction_summary.csv"
)
DIRECTION_TESTS_OUTPUT = (
    PROCESSED_DATA_DIR / "news_extreme_direction_tests.csv"
)
SENSITIVITY_OUTPUT = (
    PROCESSED_DATA_DIR / "news_extreme_threshold_sensitivity.csv"
)
EVENTS_OUTPUT = (
    PROCESSED_DATA_DIR / "news_market_events_p95.csv"
)
EVENT_STUDY_LONG_OUTPUT = (
    PROCESSED_DATA_DIR / "news_event_study_long.csv"
)
EVENT_STUDY_SUMMARY_OUTPUT = (
    PROCESSED_DATA_DIR / "news_event_study_summary.csv"
)
LEAD_LAG_OUTPUT = (
    PROCESSED_DATA_DIR / "news_market_lead_lag_correlations.csv"
)

GROUP_COMPARISON_PLOT = (
    OUTPUT_DIR / "news_extreme_vs_normal.png"
)
DIRECTION_PLOT = (
    OUTPUT_DIR / "news_by_extreme_direction.png"
)
EVENT_STUDY_PLOT = (
    OUTPUT_DIR / "news_event_study_p95.png"
)
LEAD_LAG_PLOT = (
    OUTPUT_DIR / "news_market_lead_lag.png"
)


# ---------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------

MAIN_EXTREME_FLAG = "is_extreme_p95"
EVENT_WINDOW = 5
BOOTSTRAP_ITERATIONS = 2_000
RANDOM_SEED = 42

# Episodes are separated by more than 2 * EVENT_WINDOW trading days.
# This guarantees that the [-5, +5] event-study windows do not overlap.
MAX_GAP_WITHIN_EPISODE = 2 * EVENT_WINDOW

NEWS_METRICS = {
    "all_oil": {
        "rate_column": "oil_share_per_1000",
        "count_column": "oil_headlines",
        "label": "All oil coverage",
    },
    "market_prices": {
        "rate_column": "market_price_share_per_1000",
        "count_column": "market_price_headlines",
        "label": "Market and prices",
    },
    "supply_geopolitics": {
        "rate_column": "supply_geopolitics_share_per_1000",
        "count_column": "supply_geopolitics_headlines",
        "label": "Supply and geopolitics",
    },
    "consumer_fuel": {
        "rate_column": "consumer_fuel_share_per_1000",
        "count_column": "consumer_fuel_headlines",
        "label": "Consumer fuel",
    },
}

EXTREME_FLAGS = {
    "p90": "is_extreme_p90",
    "p95": "is_extreme_p95",
    "p99": "is_extreme_p99",
}


# ---------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------


def check_file_exists(file_path: Path) -> None:
    """Verifies that the required input file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the file:\n{file_path}\n\n"
            "Run analyze_brent_volatility.py first."
        )


def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the market features and news measures."""
    print("\nLoading market and news features...")

    data = pd.read_csv(
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
        *EXTREME_FLAGS.values(),
    }

    for metric_info in NEWS_METRICS.values():
        required_columns.add(metric_info["rate_column"])
        required_columns.add(metric_info["count_column"])

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data = (
        data.sort_values("date")
        .reset_index(drop=True)
    )

    if data["date"].duplicated().any():
        raise ValueError("The input file contains duplicate dates.")

    numeric_columns = [
        "brent_price",
        "daily_return",
        "daily_return_pct",
        "absolute_return",
        "absolute_return_pct",
    ]

    for metric_info in NEWS_METRICS.values():
        numeric_columns.extend(
            [
                metric_info["rate_column"],
                metric_info["count_column"],
            ]
        )

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    for flag_column in EXTREME_FLAGS.values():
        # Works whether the CSV contains True/False strings or 0/1 values.
        if data[flag_column].dtype == bool:
            continue

        normalized = (
            data[flag_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        data[flag_column] = normalized.map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )

        if data[flag_column].isna().any():
            raise ValueError(
                f"Column {flag_column} contains invalid Boolean values."
            )

        data[flag_column] = data[flag_column].astype(bool)

    # The first trading day has no return and is excluded from tests.
    valid_rows = data["daily_return"].notna()
    invalid_return_count = int((~valid_rows).sum())

    if invalid_return_count:
        print(
            "Rows without a daily return excluded from inferential analyses: "
            f"{invalid_return_count:,}"
        )

    print(f"Rows loaded: {len(data):,}")
    print(
        "Date range: "
        f"{data['date'].min().date()} to {data['date'].max().date()}"
    )

    return data


# ---------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------


def safe_ratio(numerator: float, denominator: float) -> float:
    """Returns a ratio, while handling a zero denominator."""
    if denominator == 0:
        return np.nan
    return numerator / denominator


def bootstrap_mean_difference(
    first: np.ndarray,
    second: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    Calculates a percentile-bootstrap 95% confidence interval for:
    mean(first) - mean(second).
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]

    if len(first) == 0 or len(second) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        first_sample = rng.choice(
            first,
            size=len(first),
            replace=True,
        )
        second_sample = rng.choice(
            second,
            size=len(second),
            replace=True,
        )
        differences[iteration] = (
            first_sample.mean() - second_sample.mean()
        )

    lower, upper = np.quantile(
        differences,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Applies the Benjamini-Hochberg false-discovery-rate correction."""
    adjusted = pd.Series(
        np.nan,
        index=p_values.index,
        dtype=float,
    )

    valid = p_values.dropna().astype(float)

    if valid.empty:
        return adjusted

    ordered = valid.sort_values()
    number_of_tests = len(ordered)
    ranks = np.arange(1, number_of_tests + 1)

    raw_adjusted = (
        ordered.to_numpy()
        * number_of_tests
        / ranks
    )

    # Ensure monotonic adjusted values when moving from high to low rank.
    monotonic = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    monotonic = np.clip(monotonic, 0, 1)

    adjusted.loc[ordered.index] = monotonic

    return adjusted


# ---------------------------------------------------------
# Trading-day classification
# ---------------------------------------------------------


def add_analysis_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Adds the main P95 market-day groups used in the analysis."""
    analysis_data = data.copy()

    analysis_data["market_day_group"] = "normal"

    extreme_increase = (
        analysis_data[MAIN_EXTREME_FLAG]
        & (analysis_data["daily_return"] > 0)
    )

    extreme_decrease = (
        analysis_data[MAIN_EXTREME_FLAG]
        & (analysis_data["daily_return"] < 0)
    )

    analysis_data.loc[
        extreme_increase,
        "market_day_group",
    ] = "extreme_increase"

    analysis_data.loc[
        extreme_decrease,
        "market_day_group",
    ] = "extreme_decrease"

    analysis_data.loc[
        analysis_data["daily_return"].isna(),
        "market_day_group",
    ] = "not_available"

    return analysis_data


# ---------------------------------------------------------
# Extreme versus normal comparison
# ---------------------------------------------------------


def compare_extreme_and_normal_days(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compares P95 extreme days with non-extreme trading days.

    Two complementary tests are used:
    - Mann-Whitney U for the full news-rate distribution;
    - Fisher's exact test for the probability of any coverage.
    """
    print("\nComparing P95 extreme days with normal trading days...")

    valid = data[data["daily_return"].notna()].copy()
    extreme_mask = valid[MAIN_EXTREME_FLAG]

    rows: list[dict[str, float | int | str]] = []

    for metric_name, metric_info in NEWS_METRICS.items():
        rate_column = metric_info["rate_column"]
        count_column = metric_info["count_column"]

        extreme_rates = (
            valid.loc[extreme_mask, rate_column]
            .dropna()
            .to_numpy(dtype=float)
        )

        normal_rates = (
            valid.loc[~extreme_mask, rate_column]
            .dropna()
            .to_numpy(dtype=float)
        )

        extreme_counts = (
            valid.loc[extreme_mask, count_column]
            .fillna(0)
            .to_numpy(dtype=float)
        )

        normal_counts = (
            valid.loc[~extreme_mask, count_column]
            .fillna(0)
            .to_numpy(dtype=float)
        )

        extreme_any = int((extreme_counts > 0).sum())
        normal_any = int((normal_counts > 0).sum())

        extreme_none = int(len(extreme_counts) - extreme_any)
        normal_none = int(len(normal_counts) - normal_any)

        if len(extreme_rates) and len(normal_rates):
            mann_whitney = mannwhitneyu(
                extreme_rates,
                normal_rates,
                alternative="two-sided",
                method="auto",
            )
            mw_statistic = float(mann_whitney.statistic)
            mw_p_value = float(mann_whitney.pvalue)
        else:
            mw_statistic = np.nan
            mw_p_value = np.nan

        fisher_odds_ratio, fisher_p_value = fisher_exact(
            [
                [extreme_any, extreme_none],
                [normal_any, normal_none],
            ],
            alternative="two-sided",
        )

        extreme_mean = float(np.mean(extreme_rates))
        normal_mean = float(np.mean(normal_rates))
        mean_difference = extreme_mean - normal_mean

        ci_lower, ci_upper = bootstrap_mean_difference(
            first=extreme_rates,
            second=normal_rates,
            seed=RANDOM_SEED,
        )

        rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_info["label"],
                "rate_unit": "headlines_per_1000_total_headlines",
                "n_extreme_days": len(extreme_rates),
                "n_normal_days": len(normal_rates),
                "extreme_mean_rate": extreme_mean,
                "normal_mean_rate": normal_mean,
                "mean_difference": mean_difference,
                "mean_ratio": safe_ratio(
                    extreme_mean,
                    normal_mean,
                ),
                "extreme_median_rate": float(
                    np.median(extreme_rates)
                ),
                "normal_median_rate": float(
                    np.median(normal_rates)
                ),
                "bootstrap_ci_95_lower": ci_lower,
                "bootstrap_ci_95_upper": ci_upper,
                "mann_whitney_u": mw_statistic,
                "mann_whitney_p": mw_p_value,
                "extreme_days_with_any_coverage": extreme_any,
                "normal_days_with_any_coverage": normal_any,
                "extreme_any_coverage_rate": safe_ratio(
                    extreme_any,
                    len(extreme_counts),
                ),
                "normal_any_coverage_rate": safe_ratio(
                    normal_any,
                    len(normal_counts),
                ),
                "coverage_odds_ratio": float(fisher_odds_ratio),
                "fisher_exact_p": float(fisher_p_value),
            }
        )

    comparison = pd.DataFrame(rows)

    comparison["mann_whitney_fdr_p"] = benjamini_hochberg(
        comparison["mann_whitney_p"]
    )

    comparison["fisher_exact_fdr_p"] = benjamini_hochberg(
        comparison["fisher_exact_p"]
    )

    return comparison


# ---------------------------------------------------------
# Increase versus decrease comparison
# ---------------------------------------------------------


def summarize_by_market_direction(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarizes normal days, extreme increases and extreme decreases.
    Also applies a Kruskal-Wallis test across the three groups.
    """
    print("\nComparing normal days, extreme increases and extreme decreases...")

    valid_groups = [
        "normal",
        "extreme_increase",
        "extreme_decrease",
    ]

    valid = data[
        data["market_day_group"].isin(valid_groups)
    ].copy()

    summary_rows: list[dict[str, float | int | str]] = []
    test_rows: list[dict[str, float | int | str]] = []

    for metric_name, metric_info in NEWS_METRICS.items():
        rate_column = metric_info["rate_column"]
        count_column = metric_info["count_column"]

        test_arrays = []

        for group_name in valid_groups:
            group = valid[
                valid["market_day_group"] == group_name
            ]

            rates = group[rate_column].dropna()
            counts = group[count_column].fillna(0)

            summary_rows.append(
                {
                    "metric": metric_name,
                    "metric_label": metric_info["label"],
                    "market_day_group": group_name,
                    "n_days": len(group),
                    "mean_rate": float(rates.mean()),
                    "median_rate": float(rates.median()),
                    "standard_deviation": float(rates.std(ddof=1)),
                    "days_with_any_coverage": int((counts > 0).sum()),
                    "any_coverage_rate": float((counts > 0).mean()),
                }
            )

            test_arrays.append(rates.to_numpy(dtype=float))

        if all(len(array) > 0 for array in test_arrays):
            test_result = kruskal(
                *test_arrays,
                nan_policy="omit",
            )

            statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)
        else:
            statistic = np.nan
            p_value = np.nan

        test_rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_info["label"],
                "test": "Kruskal-Wallis",
                "statistic": statistic,
                "p_value": p_value,
            }
        )

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)

    tests["fdr_p_value"] = benjamini_hochberg(
        tests["p_value"]
    )

    return summary, tests


# ---------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------


def create_threshold_sensitivity_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Repeats the descriptive comparison for P90, P95 and P99."""
    print("\nRunning P90/P95/P99 sensitivity analysis...")

    valid = data[data["daily_return"].notna()].copy()
    rows: list[dict[str, float | int | str]] = []

    for threshold_name, flag_column in EXTREME_FLAGS.items():
        extreme_mask = valid[flag_column]

        for metric_name, metric_info in NEWS_METRICS.items():
            rate_column = metric_info["rate_column"]
            count_column = metric_info["count_column"]

            extreme_rates = valid.loc[extreme_mask, rate_column]
            normal_rates = valid.loc[~extreme_mask, rate_column]

            extreme_counts = valid.loc[extreme_mask, count_column]
            normal_counts = valid.loc[~extreme_mask, count_column]

            rows.append(
                {
                    "threshold": threshold_name,
                    "flag_column": flag_column,
                    "metric": metric_name,
                    "metric_label": metric_info["label"],
                    "n_extreme_days": int(extreme_mask.sum()),
                    "n_non_extreme_days": int((~extreme_mask).sum()),
                    "extreme_mean_rate": float(extreme_rates.mean()),
                    "non_extreme_mean_rate": float(normal_rates.mean()),
                    "mean_difference": float(
                        extreme_rates.mean() - normal_rates.mean()
                    ),
                    "mean_ratio": safe_ratio(
                        float(extreme_rates.mean()),
                        float(normal_rates.mean()),
                    ),
                    "extreme_any_coverage_rate": float(
                        (extreme_counts > 0).mean()
                    ),
                    "non_extreme_any_coverage_rate": float(
                        (normal_counts > 0).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Independent P95 episodes and event study
# ---------------------------------------------------------


def identify_independent_extreme_episodes(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Groups nearby P95 extreme days into independent episodes.

    Extreme days separated by at most 2 * EVENT_WINDOW trading days
    belong to the same episode. The event anchor is the day with the
    largest absolute return inside that episode. This prevents overlap
    between the [-5, +5] windows used later.
    """
    print("\nIdentifying independent P95 extreme-market episodes...")

    valid = data[data["daily_return"].notna()].copy()
    extreme_positions = np.flatnonzero(
        valid[MAIN_EXTREME_FLAG].to_numpy(dtype=bool)
    )

    if len(extreme_positions) == 0:
        raise ValueError("No P95 extreme days were found.")

    episode_position_groups: list[list[int]] = []
    current_group = [int(extreme_positions[0])]

    for position in extreme_positions[1:]:
        position = int(position)

        if position - current_group[-1] <= MAX_GAP_WITHIN_EPISODE:
            current_group.append(position)
        else:
            episode_position_groups.append(current_group)
            current_group = [position]

    episode_position_groups.append(current_group)

    episode_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for event_id, positions in enumerate(
        episode_position_groups,
        start=1,
    ):
        episode_extremes = valid.iloc[positions].copy()

        anchor_index_label = episode_extremes[
            "absolute_return"
        ].idxmax()

        anchor = valid.loc[anchor_index_label]
        anchor_position = int(
            valid.index.get_loc(anchor_index_label)
        )

        first_position = positions[0]
        last_position = positions[-1]

        event_window_complete = (
            anchor_position - EVENT_WINDOW >= 0
            and anchor_position + EVENT_WINDOW < len(valid)
        )

        episode_rows.append(
            {
                "event_id": event_id,
                "episode_first_extreme_date": valid.iloc[
                    first_position
                ]["date"],
                "episode_last_extreme_date": valid.iloc[
                    last_position
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
                "event_window_complete": bool(event_window_complete),
            }
        )

    events = pd.DataFrame(episode_rows)

    complete_count = int(events["event_window_complete"].sum())

    print(f"Independent episodes identified: {len(events):,}")
    print(f"Episodes with complete ±{EVENT_WINDOW} window: {complete_count:,}")

    return events


def build_event_study_long_table(
    data: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds one row per event and relative trading day from -5 to +5.
    """
    valid = (
        data[data["daily_return"].notna()]
        .reset_index(drop=True)
        .copy()
    )

    complete_events = events[
        events["event_window_complete"]
    ].copy()

    event_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for _, event in complete_events.iterrows():
        anchor_position = int(event["anchor_position"])

        for relative_day in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
            observation = valid.iloc[anchor_position + relative_day]

            row: dict[str, float | int | str | pd.Timestamp] = {
                "event_id": int(event["event_id"]),
                "anchor_date": event["anchor_date"],
                "anchor_direction": event["anchor_direction"],
                "anchor_absolute_return_pct": float(
                    event["anchor_absolute_return_pct"]
                ),
                "relative_trading_day": relative_day,
                "observation_date": observation["date"],
                "observation_brent_price": float(
                    observation["brent_price"]
                ),
            }

            for metric_name, metric_info in NEWS_METRICS.items():
                row[f"{metric_name}_rate"] = float(
                    observation[metric_info["rate_column"]]
                )
                row[f"{metric_name}_count"] = int(
                    observation[metric_info["count_column"]]
                )

            event_rows.append(row)

    event_long = pd.DataFrame(event_rows)

    # Event-specific pre-event baseline: trading days -5 through -2.
    baseline_window = event_long[
        event_long["relative_trading_day"].between(
            -EVENT_WINDOW,
            -2,
        )
    ]

    for metric_name in NEWS_METRICS:
        rate_column = f"{metric_name}_rate"
        baseline_column = f"{metric_name}_pre_event_baseline"
        excess_column = f"{metric_name}_excess_over_baseline"

        baseline = (
            baseline_window
            .groupby("event_id")[rate_column]
            .mean()
            .rename(baseline_column)
        )

        event_long = event_long.merge(
            baseline,
            left_on="event_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )

        event_long[excess_column] = (
            event_long[rate_column]
            - event_long[baseline_column]
        )

    return event_long


def summarize_event_study(
    event_long: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregates event-window news rates and excess over baseline."""
    rows: list[dict[str, float | int | str]] = []

    for metric_name, metric_info in NEWS_METRICS.items():
        rate_column = f"{metric_name}_rate"
        excess_column = f"{metric_name}_excess_over_baseline"

        for relative_day, group in event_long.groupby(
            "relative_trading_day"
        ):
            rates = group[rate_column].dropna()
            excess = group[excess_column].dropna()

            n_events = len(rates)
            standard_error = (
                float(rates.std(ddof=1) / np.sqrt(n_events))
                if n_events > 1
                else np.nan
            )

            rows.append(
                {
                    "metric": metric_name,
                    "metric_label": metric_info["label"],
                    "relative_trading_day": int(relative_day),
                    "n_events": n_events,
                    "mean_rate": float(rates.mean()),
                    "median_rate": float(rates.median()),
                    "standard_error": standard_error,
                    "normal_approx_ci_95_lower": float(
                        rates.mean() - 1.96 * standard_error
                    ) if np.isfinite(standard_error) else np.nan,
                    "normal_approx_ci_95_upper": float(
                        rates.mean() + 1.96 * standard_error
                    ) if np.isfinite(standard_error) else np.nan,
                    "mean_excess_over_pre_event_baseline": float(
                        excess.mean()
                    ),
                    "median_excess_over_pre_event_baseline": float(
                        excess.median()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Lead-lag correlation
# ---------------------------------------------------------


def calculate_lead_lag_correlations(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates Spearman correlations between today's absolute return
    and news coverage on trading days -5 through +5.

    A negative relative day means news precedes the market movement.
    A positive relative day means news follows the market movement.
    """
    print("\nCalculating lead-lag Spearman correlations...")

    valid = data[data["daily_return"].notna()].copy()
    rows: list[dict[str, float | int | str]] = []

    for metric_name, metric_info in NEWS_METRICS.items():
        rate_column = metric_info["rate_column"]

        for relative_day in range(-EVENT_WINDOW, EVENT_WINDOW + 1):
            # Align news at t + relative_day with absolute return at t.
            shifted_news = valid[rate_column].shift(-relative_day)

            pairs = pd.DataFrame(
                {
                    "absolute_return": valid["absolute_return"],
                    "news_rate": shifted_news,
                }
            ).dropna()

            if len(pairs) >= 3:
                correlation = spearmanr(
                    pairs["absolute_return"],
                    pairs["news_rate"],
                )
                rho = float(correlation.statistic)
                p_value = float(correlation.pvalue)
            else:
                rho = np.nan
                p_value = np.nan

            rows.append(
                {
                    "metric": metric_name,
                    "metric_label": metric_info["label"],
                    "relative_trading_day": relative_day,
                    "interpretation": (
                        "news_before_market"
                        if relative_day < 0
                        else "same_day"
                        if relative_day == 0
                        else "news_after_market"
                    ),
                    "n_pairs": len(pairs),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    correlations = pd.DataFrame(rows)

    correlations["fdr_p_value"] = (
        correlations
        .groupby("metric", group_keys=False)["p_value"]
        .apply(benjamini_hochberg)
    )

    return correlations


# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------


def plot_extreme_vs_normal(comparison: pd.DataFrame) -> None:
    """Plots mean news rates on P95 extreme and normal days."""
    labels = comparison["metric_label"].tolist()
    normal_means = comparison["normal_mean_rate"].to_numpy()
    extreme_means = comparison["extreme_mean_rate"].to_numpy()

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

    ax.set_title("Oil-News Coverage on Extreme and Normal Brent Trading Days")
    ax.set_xlabel("News category")
    ax.set_ylabel("Mean headlines per 1,000 ABC headlines")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        GROUP_COMPARISON_PLOT,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_direction_summary(summary: pd.DataFrame) -> None:
    """Plots news rates by normal/increase/decrease market-day group."""
    metric_order = list(NEWS_METRICS.keys())
    group_order = [
        "normal",
        "extreme_increase",
        "extreme_decrease",
    ]

    labels = [
        NEWS_METRICS[metric]["label"]
        for metric in metric_order
    ]

    positions = np.arange(len(metric_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))

    for group_index, group_name in enumerate(group_order):
        group_values = (
            summary[
                summary["market_day_group"] == group_name
            ]
            .set_index("metric")
            .reindex(metric_order)["mean_rate"]
            .to_numpy()
        )

        offset = (group_index - 1) * width

        ax.bar(
            positions + offset,
            group_values,
            width,
            label=group_name.replace("_", " ").title(),
        )

    ax.set_title("Oil-News Coverage by Direction of Extreme Brent Movement")
    ax.set_xlabel("News category")
    ax.set_ylabel("Mean headlines per 1,000 ABC headlines")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        DIRECTION_PLOT,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_event_study(summary: pd.DataFrame) -> None:
    """Plots average excess news coverage around P95 episodes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for metric_name, metric_info in NEWS_METRICS.items():
        metric_data = (
            summary[summary["metric"] == metric_name]
            .sort_values("relative_trading_day")
        )

        ax.plot(
            metric_data["relative_trading_day"],
            metric_data["mean_excess_over_pre_event_baseline"],
            marker="o",
            label=metric_info["label"],
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="Extreme-market event",
    )
    ax.axhline(0, linewidth=1.0)

    ax.set_title("Oil-News Coverage Around Independent P95 Market Episodes")
    ax.set_xlabel("Trading days relative to event anchor")
    ax.set_ylabel(
        "Mean excess headlines per 1,000\n"
        "relative to each event's pre-event baseline"
    )
    ax.set_xticks(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        EVENT_STUDY_PLOT,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_lead_lag_correlations(correlations: pd.DataFrame) -> None:
    """Plots lead-lag Spearman correlations."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for metric_name, metric_info in NEWS_METRICS.items():
        metric_data = (
            correlations[correlations["metric"] == metric_name]
            .sort_values("relative_trading_day")
        )

        ax.plot(
            metric_data["relative_trading_day"],
            metric_data["spearman_rho"],
            marker="o",
            label=metric_info["label"],
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="Same trading day",
    )
    ax.axhline(0, linewidth=1.0)

    ax.set_title("Lead-Lag Relationship Between Brent Movements and Oil News")
    ax.set_xlabel(
        "Relative trading day of news\n"
        "negative = news before market movement; positive = after"
    )
    ax.set_ylabel("Spearman correlation with absolute daily return")
    ax.set_xticks(range(-EVENT_WINDOW, EVENT_WINDOW + 1))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        LEAD_LAG_PLOT,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------
# Output and console summary
# ---------------------------------------------------------


def save_outputs(
    analysis_data: pd.DataFrame,
    comparison: pd.DataFrame,
    direction_summary: pd.DataFrame,
    direction_tests: pd.DataFrame,
    sensitivity: pd.DataFrame,
    events: pd.DataFrame,
    event_long: pd.DataFrame,
    event_summary: pd.DataFrame,
    lead_lag: pd.DataFrame,
) -> None:
    """Saves all data outputs."""
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_data.to_csv(
        ANALYSIS_DATA_OUTPUT,
        index=False,
        date_format="%Y-%m-%d",
    )
    comparison.to_csv(
        GROUP_COMPARISON_OUTPUT,
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
    sensitivity.to_csv(
        SENSITIVITY_OUTPUT,
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
    lead_lag.to_csv(
        LEAD_LAG_OUTPUT,
        index=False,
    )


def print_summary(
    analysis_data: pd.DataFrame,
    comparison: pd.DataFrame,
    direction_summary: pd.DataFrame,
    events: pd.DataFrame,
    lead_lag: pd.DataFrame,
) -> None:
    """Prints the main results needed for the next review step."""
    print("\n" + "=" * 72)
    print("NEWS-MARKET RELATIONSHIP SUMMARY")
    print("=" * 72)

    group_counts = (
        analysis_data["market_day_group"]
        .value_counts()
    )

    print("\nTrading-day groups:")
    print(group_counts.to_string())

    display_columns = [
        "metric_label",
        "extreme_mean_rate",
        "normal_mean_rate",
        "mean_difference",
        "mean_ratio",
        "mann_whitney_fdr_p",
        "extreme_any_coverage_rate",
        "normal_any_coverage_rate",
        "fisher_exact_fdr_p",
    ]

    print("\nP95 extreme versus normal comparison:")
    print(
        comparison[display_columns].to_string(
            index=False,
            formatters={
                "extreme_mean_rate": lambda value: f"{value:.3f}",
                "normal_mean_rate": lambda value: f"{value:.3f}",
                "mean_difference": lambda value: f"{value:.3f}",
                "mean_ratio": lambda value: f"{value:.2f}",
                "mann_whitney_fdr_p": lambda value: f"{value:.4g}",
                "extreme_any_coverage_rate": lambda value: f"{value:.2%}",
                "normal_any_coverage_rate": lambda value: f"{value:.2%}",
                "fisher_exact_fdr_p": lambda value: f"{value:.4g}",
            },
        )
    )

    print("\nMean news rates by market-day group:")
    direction_pivot = direction_summary.pivot(
        index="metric_label",
        columns="market_day_group",
        values="mean_rate",
    )
    print(direction_pivot.round(3).to_string())

    print(
        "\nIndependent non-overlapping P95 episodes: "
        f"{len(events):,}"
    )

    strongest_events = (
        events.sort_values(
            "anchor_absolute_return_pct",
            ascending=False,
        )
        .head(10)
        [
            [
                "event_id",
                "anchor_date",
                "anchor_daily_return_pct",
                "anchor_direction",
                "n_p95_extreme_days_in_episode",
            ]
        ]
    )

    print("\nTen strongest independent episodes:")
    print(
        strongest_events.to_string(
            index=False,
            formatters={
                "anchor_daily_return_pct": lambda value: f"{value:.2f}%",
            },
        )
    )

    strongest_correlations = (
        lead_lag.assign(
            absolute_rho=lead_lag["spearman_rho"].abs()
        )
        .sort_values("absolute_rho", ascending=False)
        .head(10)
        [
            [
                "metric_label",
                "relative_trading_day",
                "spearman_rho",
                "fdr_p_value",
            ]
        ]
    )

    print("\nLargest lead-lag correlations:")
    print(
        strongest_correlations.to_string(
            index=False,
            formatters={
                "spearman_rho": lambda value: f"{value:.3f}",
                "fdr_p_value": lambda value: f"{value:.4g}",
            },
        )
    )


def main() -> None:
    """Runs the complete news-market relationship analysis."""
    print("=" * 72)
    print("GeoOil-Pulse: News-Market Relationship Analysis")
    print("=" * 72)

    check_file_exists(INPUT_FILE)

    data = load_data(INPUT_FILE)
    analysis_data = add_analysis_groups(data)

    comparison = compare_extreme_and_normal_days(
        analysis_data
    )

    direction_summary, direction_tests = (
        summarize_by_market_direction(
            analysis_data
        )
    )

    sensitivity = create_threshold_sensitivity_summary(
        analysis_data
    )

    events = identify_independent_extreme_episodes(
        analysis_data
    )

    event_long = build_event_study_long_table(
        data=analysis_data,
        events=events,
    )

    event_summary = summarize_event_study(
        event_long
    )

    lead_lag = calculate_lead_lag_correlations(
        analysis_data
    )

    save_outputs(
        analysis_data=analysis_data,
        comparison=comparison,
        direction_summary=direction_summary,
        direction_tests=direction_tests,
        sensitivity=sensitivity,
        events=events,
        event_long=event_long,
        event_summary=event_summary,
        lead_lag=lead_lag,
    )

    plot_extreme_vs_normal(comparison)
    plot_direction_summary(direction_summary)
    plot_event_study(event_summary)
    plot_lead_lag_correlations(lead_lag)

    print_summary(
        analysis_data=analysis_data,
        comparison=comparison,
        direction_summary=direction_summary,
        events=events,
        lead_lag=lead_lag,
    )

    print("\nSaved processed files:")
    print(f"1. {ANALYSIS_DATA_OUTPUT}")
    print(f"2. {GROUP_COMPARISON_OUTPUT}")
    print(f"3. {DIRECTION_SUMMARY_OUTPUT}")
    print(f"4. {DIRECTION_TESTS_OUTPUT}")
    print(f"5. {SENSITIVITY_OUTPUT}")
    print(f"6. {EVENTS_OUTPUT}")
    print(f"7. {EVENT_STUDY_LONG_OUTPUT}")
    print(f"8. {EVENT_STUDY_SUMMARY_OUTPUT}")
    print(f"9. {LEAD_LAG_OUTPUT}")

    print("\nSaved plots:")
    print(f"1. {GROUP_COMPARISON_PLOT}")
    print(f"2. {DIRECTION_PLOT}")
    print(f"3. {EVENT_STUDY_PLOT}")
    print(f"4. {LEAD_LAG_PLOT}")

    print("\nNews-market relationship analysis completed successfully.")


if __name__ == "__main__":
    main()
# The Ripple Effect: Oil Market Movements and News Coverage (GeoOil-Pulse)

**Course:** 67978 - A Needle in a Data Haystack  
**Hebrew University of Jerusalem**  

---

## Project Overview
This project investigates the temporal and lead-lag relationship between daily Brent crude oil price shocks and media news coverage using historical spot prices (EIA) and over 1.2 million news headlines (ABC News Australia) from 2003 to 2021.

Key analyses include:
- Normalized oil-news volume on extreme-return days (P90, P95, P99).
- Independent episode Event Studies ([-5, +5] trading-day windows).
- Spearman Lead-Lag cross-correlations.
- Lexicon-based sentiment scoring using VADER, accompanied by manual ground-truth validation ($N=150$).

---

## Repository Structure
```text
├── data/
│   ├── raw/                  # Raw input datasets (rbrted.xls, abcnews-date-text.csv)
│   └── processed/            # Cleaned data, extracted market features, and event tables
├── output/                   # Generated evaluation plots and figures (.png)
├── src/                      # Analysis and modeling scripts
│   ├── prepare_data.py
│   ├── filter_oil_headlines.py
│   ├── analyze_brent_volatility.py
│   ├── analyze_news_market_relationship.py
│   ├── analyze_news_sentiment.py
│   └── evaluate_vader_validation.py

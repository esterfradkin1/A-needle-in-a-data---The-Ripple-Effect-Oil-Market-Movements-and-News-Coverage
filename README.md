# The Ripple Effect: Oil Market Movements and News Coverage (GeoOil-Pulse)

**Course:** 67978 - A Needle in a Data Haystack  
**Institution:** The Hebrew University of Jerusalem  

### Authors:
| Full Name | ID | CSE Username | University Email Address |
| :--- | :--- | :--- | :--- |
| **Gaya Cohen** | 329075543 | gaya_cohen.33 | gaya.cohen@mail.huji.ac.il |
| **Nikol Fedorovsky** | 327569505 | Nikol.fed | nikol.fedorovsky@mail.huji.ac.il |
| **Ester Fradkin** | 208138735 | esterfradkin | ester.fradkin@mail.huji.ac.il |

---

## Project Overview
This project investigates the temporal and lead-lag relationship between daily Brent crude oil price shocks and media news coverage using historical spot prices (EIA) and over 1.2 million news headlines (ABC News Australia) from 2003 to 2021.

### Key Analyses:
- **Coverage Volume Disparities:** Normalized oil-news volume on extreme-return days (P90, P95, P99) compared to normal trading days.
- **Event Study:** Independent extreme-market episodes analyzed over a symmetric $[-5, +5]$ trading-day window.
- **Lead-Lag Dynamics:** Non-parametric Spearman cross-correlations across temporal shifts.
- **Sentiment Analysis & Ground-Truth Validation:** Lexicon-based sentiment scoring via VADER with domain-specific manual validation ($N=150$).

---

## Repository Structure
```text
├── data/
│   ├── raw/                          # Raw input datasets
│   │   ├── abcnews-date-text.zip     # ABC News dataset archive (extract to abcnews-date-text.csv)
│   │   ├── abcnews-date-text.csv     # Extracted 1.2M ABC headlines
│   │   └── rbrted.xls                # Daily Brent crude oil historical spot prices (EIA)
│   └── processed/                    # Cleaned data, extracted market features, and event tables
├── output/                           # Generated evaluation figures (.png) and review samples
├── src/                              # Analysis and modeling pipeline
│   ├── prepare_data.py               # Step 1: Initial data cleaning and temporal alignment
│   ├── filter_oil_headlines.py       # Step 2: Domain-specific regex filtering and categorization
│   ├── analyze_brent_volatility.py   # Step 3: Returns, rolling volatility, and P95 extreme flags
│   ├── analyze_news_market_relationship.py # Step 4: Group comparisons, event study, and lead-lag
│   ├── analyze_news_sentiment.py     # Step 5: VADER scoring and daily aggregation
│   └── evaluate_vader_validation.py  # Step 6: Manual ground-truth validation & confusion matrix


## Installation & Setup

### Clone the repository

### Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

### Install required packages
pip install pandas numpy matplotlib scipy nltk scikit-learn xlrd openpyxl

### Execution Pipeline
# Step 1: Clean and align raw EIA and ABC News datasets
python src/prepare_data.py

# Step 2: Filter and categorize oil-related headlines
python src/filter_oil_headlines.py

# Step 3: Compute daily returns, rolling volatility, and extreme-day thresholds
python src/analyze_brent_volatility.py

# Step 4: Run market-relationship comparisons, event studies, and lead-lag analysis
python src/analyze_news_market_relationship.py

# Step 5: Perform VADER sentiment scoring on categorized headlines
python src/analyze_news_sentiment.py

# Step 6: Evaluate VADER performance against ground-truth labels and generate Confusion Matrix
python src/evaluate_vader_validation.py

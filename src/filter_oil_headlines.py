from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'output'

ABC_RAW_FILE = RAW_DATA_DIR / 'abcnews-date-text.csv'
ABC_DAILY_COUNTS_FILE = PROCESSED_DATA_DIR / 'abc_daily_counts.csv'
ABC_BRENT_DAILY_FILE = PROCESSED_DATA_DIR / 'abc_brent_daily.csv'

FILTERED_HEADLINES_OUTPUT = PROCESSED_DATA_DIR / 'oil_headlines_categorized.csv'
OIL_DAILY_OUTPUT = PROCESSED_DATA_DIR / 'abc_oil_daily.csv'
FINAL_MERGED_OUTPUT = PROCESSED_DATA_DIR / 'abc_brent_oil_daily.csv'
MATCHED_SAMPLE_OUTPUT = OUTPUT_DIR / 'sample_matched_oil_headlines.csv'
REJECTED_SAMPLE_OUTPUT = OUTPUT_DIR / 'sample_rejected_oil_candidates.csv'
MULTI_CATEGORY_SAMPLE_OUTPUT = OUTPUT_DIR / 'sample_multi_category_headlines.csv'

GLOBAL_EXCLUSION_PATTERNS = {
    'olive_oil': r'\bolive\s+oil\b',
    'cooking_oil': r'\bcooking\s+oil\b',
    'vegetable_oil': r'\bvegetable\s+oil\b',
    'essential_oil': r'\bessential\s+oils?\b',
    'fish_oil': r'\bfish\s+oil\b',
    'coconut_oil': r'\bcoconut\s+oil\b',
    'palm_oil': r'\bpalm\s+oil\b',
    'avocado_oil': r'\bavocado\s+oil\b',
    'canola_oil': r'\bcanola\s+oil\b',
    'sunflower_oil': r'\bsunflower\s+oil\b',
    'sesame_oil': r'\bsesame\s+oil\b',
    'tea_tree_oil': r'\btea\s+tree\s+oil\b',
    'baby_oil': r'\bbaby\s+oil\b',
    'hair_oil': r'\bhair\s+oil\b',
    'motor_oil': r'\bmotor\s+oil\b',
    'engine_oil': r'\bengine\s+oil\b',
    'lubricating_oil': r'\blubricating\s+oil\b',
    'oil_painting': r'\boil\s+paintings?\b',
    'oil_on_canvas': r'\boil\s+on\s+canvas\b',
    'snake_oil': r'\bsnake\s+oil\b',
    'petroleum_jelly': r'\bpetroleum\s+jelly\b',
    'petrol_sniffing': (
        r'\b(?:petrol|fuel)\b.*\b(?:sniff\w*|unsniffable)\b|'
        r'\b(?:sniff\w*|unsniffable)\b.*\b(?:petrol|fuel)\b'
    ),
    'cannabis_oil': r'\b(?:cannabis|hemp|cbd)\s+oil\b',
}

MARKET_PRICE_PATTERNS = {
    # Brent is accepted only near explicit market/oil terminology.
    'brent_market_context': (
        r'\bbrent\b(?:\s+[a-z0-9]+){0,3}\s+'
        r'\b(?:crude|oil|prices?|futures|benchmark|market)\b|'
        r'\b(?:crude|oil|prices?|futures|benchmark|market)\b'
        r'(?:\s+[a-z0-9]+){0,3}\s+\bbrent\b'
    ),
    'wti': r'\bwti\b|\bwest\s+texas\s+intermediate\b',
    'crude_oil': r'\bcrude\s+oil\b|\bcrude\s+(?:prices?|futures?|market)\b',
    'oil_price': r'\boil\s+prices?\b|\bprices?\s+of\s+oil\b',
    'oil_market': r'\boil\s+(?:markets?|futures?|trading|stocks?)\b',
    'oil_price_movement': (
        r'\boil\s+(?:rises?|falls?|drops?|jumps?|slides?|gains?|climbs?|surges?|'
        r'slumps?|rebounds?|plunges?|soars?)\b|'
        r'\boil\s+(?:hits?|reaches?|tops?|above|below|near)\b.*'
        r'\b(?:record|high|low|usd|dollars?|barrel)\b'
    ),
    'oil_crisis_or_fears': (
        r'\boil\s+(?:crisis|shock|fears?|rally|selloff|sell\s+off)\b'
    ),
    'oil_barrel_price': (
        r'\b(?:oil|crude)\b.*\b(?:usd\s*)?\d+(?:\.\d+)?\s+(?:a|per)\s+barrel\b|'
        r'\b(?:usd\s*)?\d+(?:\.\d+)?\s+(?:a|per)\s+barrel\b.*\b(?:oil|crude)\b'
    ),
    'oil_and_financial_market': (
        r'\b(?:wall\s+street|share\s+market|stock\s+market|stocks?|shares?|markets?|inflation)\b'
        r'.*\boil\b|'
        r'\boil\b.*\b(?:wall\s+street|share\s+market|stock\s+market|stocks?|shares?|markets?|inflation)\b'
    ),
}

SUPPLY_GEOPOLITICS_PATTERNS = {
    'opec': r'\bopec(?:\+)?\b',
    'petroleum_context': (
        r'\bpetroleum\s+(?:industry|sector|exploration|production|prices?|reserves?|'
        r'exports?|imports?|projects?|licen[cs]es?|permits?|potential|wells?)\b|'
        r'\b(?:industry|sector|exploration|production|prices?|reserves?|exports?|'
        r'imports?|projects?|licen[cs]es?|permits?|potential|wells?)\s+petroleum\b'
    ),
    'shale_oil': r'\bshale\s+oil\b',
    'oil_core_supply': (
        r'\b(?:oil|crude)\s+(?:production|output|supply|demand|exports?|imports?|reserves?)\b|'
        r'\b(?:production|output|supply|exports?|imports?)\s+of\s+(?:oil|crude)\b|'
        r'\bdemand\s+for\s+(?:oil|crude)\b|'
        r'\b(?:oil|crude)\s+producing\b'
    ),
    'oil_exploration_drilling': (
        r'\b(?:oil|crude)\s+(?:exploration|drilling|licen[cs]es?|leases?|permits?|rig\s+count)\b|'
        r'\b(?:exploration|drilling|licen[cs]es?|leases?|permits?|rig\s+count)\b.*\b(?:oil|crude)\b'
    ),
    'oil_industry_projects': (
        r'\b(?:oil|crude)\s+(?:companies|company|firms?|giants?|projects?|industry|sector|depots?)\b|'
        r'\b(?:oil|crude)\s+fields?\b.*\b(?:project|development|approved|approval|green\s+light|investment|protect|troops?)\b|'
        r'\b(?:project|development|approved|approval|green\s+light|investment|protect|troops?)\b.*\b(?:oil|crude)\s+fields?\b'
    ),
    'oil_refinery_operations': (
        r'\b(?:oil\s+)?refiner(?:y|ies)\b.*\b(?:closure|closes?|closed|shutdown|shuts?|'
        r'capacity|production|output|operations?|reopens?|strike|security|supply)\b|'
        r'\b(?:closure|closes?|closed|shutdown|shuts?|capacity|production|output|operations?|'
        r'reopens?|strike|security|supply)\b.*\b(?:oil\s+)?refiner(?:y|ies)\b'
    ),
    'oil_infrastructure_disruption': (
        r'\b(?:oil|crude)\s+(?:pipeline|terminal|facility|field)\b.*'
        r'\b(?:attack\w*|sabotage|shutdown|shut\s+down|closure|closed|halt|halts|'
        r'suspend|suspends|disrupt|disruption|outage|production|output|supply|export|'
        r'largest|major|key|strategic)\b|'
        r'\b(?:attack\w*|sabotage|shutdown|shut\s+down|closure|closed|halt|halts|'
        r'suspend|suspends|disrupt|disruption|outage|production|output|supply|export|'
        r'largest|major|key|strategic)\b.*\b(?:oil|crude)\s+(?:pipeline|terminal|facility|field)\b'
    ),
    'strategic_oil_tanker': (
        r'\b(?:oil|crude)\s+tanker\b.*\b(?:attack|hijack|seiz|pirat|sanction|embargo|'
        r'blockade|strait|gulf|supply|export|route|war|terror)\w*\b|'
        r'\b(?:attack|hijack|seiz|pirat|sanction|embargo|blockade|strait|gulf|supply|'
        r'export|route|war|terror)\w*\b.*\b(?:oil|crude)\s+tanker\b'
    ),
    'fuel_shortage_security': (
        r'\b(?:petrol|diesel|gasoline|fuel)\s+(?:shortage|shortages|security|supply|strike)\b|'
        r'\b(?:shortage|shortages)\b.*\b(?:petrol|diesel|gasoline|fuel)\b|'
        r'\bruns?\s+low\s+on\s+(?:petrol|diesel|gasoline|fuel)\b'
    ),
    'oil_policy': (
        r'\b(?:oil|crude)\s+(?:sanctions?|embargo(?:es)?|policy|policies|deal|deals|talks?|'
        r'leases?|subsid(?:y|ies)|ban|bans|cuts?|tax|taxes)\b|'
        r'\b(?:sanctions?|embargo(?:es)?|policy|policies|deal|deals|talks?|leases?|'
        r'subsid(?:y|ies)|ban|bans|cuts?|tax|taxes)\b.*\b(?:oil|crude)\b'
    ),
    'oil_and_gas_industry': (
        r'\boil\s+and\s+gas\b.*\b(?:industry|sector|companies|production|exploration|'
        r'drilling|leases?|subsid(?:y|ies)|investment|projects?|exports?|policy)\b|'
        r'\b(?:industry|sector|companies|production|exploration|drilling|leases?|'
        r'subsid(?:y|ies)|investment|projects?|exports?|policy)\b.*\boil\s+and\s+gas\b'
    ),
    'oil_geopolitical_event': (
        r'\b(?:war|conflict|dispute|tension|tensions|sanction|sanctions|embargo|embargoes|'
        r'blockade|rebels?)\b.*\b(?:oil|crude\s+oil|opec)\b|'
        r'\b(?:oil|crude\s+oil|opec)\b.*\b(?:war|conflict|dispute|tension|tensions|'
        r'sanction|sanctions|embargo|embargoes|blockade|rebels?)\b|'
        r'\b(?:attack\w*|sabotage|terror)\b.*\b(?:oil|crude\s+oil)\s+'
        r'(?:pipeline|field|facility|terminal|tanker|refinery|production|industry)\b|'
        r'\b(?:oil|crude\s+oil)\s+(?:pipeline|field|facility|terminal|tanker|refinery|'
        r'production|industry)\b.*\b(?:attack\w*|sabotage|terror)\b'
    ),
    'oil_geopolitical_country': (
        r'\b(?:iran|iraq|saudi|russia|libya|venezuela|south\s+sudan|syria|middle\s+east)\b'
        r'.*\b(?:oil|crude\s+oil)\b|'
        r'\b(?:oil|crude\s+oil)\b.*\b(?:iran|iraq|saudi|russia|libya|venezuela|south\s+sudan|'
        r'middle\s+east)\b'
    ),
}

CONSUMER_FUEL_PATTERNS = {
    'fuel_price_or_cost': (
        r'\b(?:petrol|diesel|gasoline|fuel|unleaded)\s+'
        r'(?:prices?|costs?|bills?|tax|taxes|excise|surcharges?|subsid(?:y|ies)|levy|levies)\b'
    ),
    'reversed_fuel_price_or_cost': (
        r'\b(?:prices?|costs?|bills?|tax|taxes|excise|surcharges?|subsid(?:y|ies)|levy|levies)\b.*'
        r'\b(?:petrol|diesel|gasoline|fuel|unleaded)\b'
    ),
    'fuel_price_level': (
        r'\b(?:rising|higher|soaring|record|cheaper|expensive)\s+'
        r'(?:petrol|diesel|gasoline|fuel|unleaded)\b'
        r'(?!\s+(?:loads?|hazards?|moisture|vegetation))|'
        r'\b(?:petrol|diesel|gasoline|unleaded)\b.*'
        r'\b(?:high|low|record|peak|higher|lower|cheaper|expensive)\b|'
        r'\b(?:petrol|diesel|gasoline|unleaded)\s+(?:reaches?|hits?)\s+(?:[0-9]+|record|high|low)\b'
    ),
    'fuel_price_movement': (
        r'\b(?:petrol|diesel|gasoline|fuel|unleaded)\s+'
        r'(?:rises?|falls?|jumps?|drops?|soars?|surges?|climbs?|slumps?)\b'
    ),
    'fuel_cost_proximity': (
        r'\bfuel\b(?:\s+[a-z0-9]+){0,2}\s+\b(?:costs?|prices?|surcharges?)\b|'
        r'\b(?:costs?|prices?|surcharges?)\b(?:\s+[a-z0-9]+){0,2}\s+\bfuel\b'
    ),
    'pump_prices': r'\bpump\s+prices?\b',
    'petrol_at_pump': r'\bpetrol\s+(?:at|from)\s+the\s+pump\b',
}

BROAD_CANDIDATE_PATTERN = (
    r'\b(?:oil|crude|brent|wti|opec|petroleum|petrol|diesel|gasoline|fuel|unleaded)\b'
)

LOCAL_INCIDENT_PATTERN = (
    r'\b(?:spill|spills|spilled|leak|leaks|leaking|fire|blaze|collision|collides|crash|'
    r'accident|dies|died|death|deaths|killed|injured|missing|evacuat(?:e|ed|ion)|rescue|'
    r'worker|workers|crew|safety|compensation)\b'
)

MARKET_IMPACT_PATTERN = (
    r'\b(?:price|prices|market|markets|production|output|supply|export|exports|shutdown|'
    r'shut\s+down|closure|closed|disrupt|disruption|halt|halts|suspend|suspends|outage|'
    r'capacity|largest|major|key|strategic|opec|sanction|sanctions|war|attack|attacks|'
    r'sabotage|embargo|blockade)\b'
)


def check_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f'Could not find the file:\n{file_path}\n\n'
            'Make sure the input files are in data/raw and that prepare_data.py ran successfully.'
        )


def load_and_clean_abc_headlines(file_path: Path) -> pd.DataFrame:
    print('\nLoading the original ABC News dataset...')
    news = pd.read_csv(
        file_path,
        usecols=['publish_date', 'headline_text'],
        dtype={'publish_date': 'string', 'headline_text': 'string'},
        low_memory=False,
    )
    news['date'] = pd.to_datetime(news['publish_date'], format='%Y%m%d', errors='coerce')
    news['headline_text'] = news['headline_text'].str.strip()
    news = news.dropna(subset=['date', 'headline_text'])
    news = news[news['headline_text'] != '']
    news = news.drop_duplicates(subset=['date', 'headline_text'], keep='first')
    news['normalized_headline'] = (
        news['headline_text'].str.lower()
        .str.replace(r'[^a-z0-9+]+', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    news = news.reset_index(drop=True)
    print(f'Headlines loaded: {len(news):,}')
    return news


def build_reason_mask(headlines, patterns, blocked_mask=None):
    mask = pd.Series(False, index=headlines.index, dtype=bool)
    reason = pd.Series('', index=headlines.index, dtype='string')
    if blocked_mask is None:
        blocked_mask = pd.Series(False, index=headlines.index, dtype=bool)
    for name, pattern in patterns.items():
        current = headlines.str.contains(pattern, regex=True, na=False) & ~blocked_mask & ~mask
        mask |= current
        reason.loc[current] = name
    return mask, reason


def combine_category_labels(market_mask, supply_mask, consumer_mask):
    labels = pd.Series('', index=market_mask.index, dtype='string')
    for name, mask in [
        ('market_prices', market_mask),
        ('supply_geopolitics', supply_mask),
        ('consumer_fuel', consumer_mask),
    ]:
        labels.loc[mask] = labels.loc[mask].where(labels.loc[mask] == '', labels.loc[mask] + ';') + name
    return labels


def choose_primary_category(market_mask, supply_mask, consumer_mask):
    primary = pd.Series('', index=market_mask.index, dtype='string')
    primary.loc[consumer_mask] = 'consumer_fuel'
    primary.loc[supply_mask] = 'supply_geopolitics'
    primary.loc[market_mask] = 'market_prices'
    return primary


def classify_oil_headlines(news: pd.DataFrame) -> pd.DataFrame:
    print('\nClassifying oil-related headlines...')
    # Run the detailed regex rules only on broad oil/fuel candidates.
    candidate_mask = news['normalized_headline'].str.contains(
        BROAD_CANDIDATE_PATTERN, regex=True, na=False
    )
    candidates = news.loc[candidate_mask].copy().reset_index(drop=True)
    h = candidates['normalized_headline']

    exclusion_mask, exclusion_reason = build_reason_mask(h, GLOBAL_EXCLUSION_PATTERNS)
    market_mask, market_rule = build_reason_mask(h, MARKET_PRICE_PATTERNS, exclusion_mask)
    supply_mask, supply_rule = build_reason_mask(h, SUPPLY_GEOPOLITICS_PATTERNS, exclusion_mask)
    consumer_mask, consumer_rule = build_reason_mask(h, CONSUMER_FUEL_PATTERNS, exclusion_mask)

    # Remove local accidents/human-interest incidents from supply coverage unless the
    # headline also indicates market, production, policy or geopolitical impact.
    local_incident = h.str.contains(LOCAL_INCIDENT_PATTERN, regex=True, na=False)
    market_impact = h.str.contains(MARKET_IMPACT_PATTERN, regex=True, na=False)
    consumer_economic_context = h.str.contains(
        r'\b(?:price|prices|cost|costs|bill|bills|tax|taxes|excise|surcharge|surcharges|'
        r'subsidy|subsidies|levy|levies|cheap|cheaper|expensive|inflation|airfare|profit|'
        r'profits|business|tourism)\b',
        regex=True, na=False,
    )

    suppress_supply = supply_mask & local_incident & ~market_impact
    suppress_market = market_mask & local_incident & ~market_impact
    suppress_consumer = consumer_mask & local_incident & ~consumer_economic_context

    supply_mask.loc[suppress_supply] = False
    supply_rule.loc[suppress_supply] = ''
    market_mask.loc[suppress_market] = False
    market_rule.loc[suppress_market] = ''
    consumer_mask.loc[suppress_consumer] = False
    consumer_rule.loc[suppress_consumer] = ''

    suppressed_any = suppress_supply | suppress_market | suppress_consumer
    suppression_reason = pd.Series('', index=h.index, dtype='string')
    suppression_reason.loc[suppressed_any] = 'local_incident_without_market_impact'

    relevant_mask = market_mask | supply_mask | consumer_mask
    classified = candidates.copy()
    classified['is_oil_relevant'] = relevant_mask
    classified['is_market_prices'] = market_mask
    classified['is_supply_geopolitics'] = supply_mask
    classified['is_consumer_fuel'] = consumer_mask
    classified['categories'] = combine_category_labels(market_mask, supply_mask, consumer_mask)
    classified['primary_category'] = choose_primary_category(market_mask, supply_mask, consumer_mask)
    classified['market_match_rule'] = market_rule
    classified['supply_match_rule'] = supply_rule
    classified['consumer_match_rule'] = consumer_rule
    classified['is_excluded'] = exclusion_mask
    classified['exclusion_reason'] = exclusion_reason
    classified['suppression_reason'] = suppression_reason

    print(f'Broad candidates examined: {len(candidates):,}')
    print(f'Relevant oil headlines: {int(relevant_mask.sum()):,}')
    print(f'Market-price matches: {int(market_mask.sum()):,}')
    print(f'Supply-geopolitics matches: {int(supply_mask.sum()):,}')
    print(f'Consumer-fuel matches: {int(consumer_mask.sum()):,}')
    print(f'Explicitly excluded headlines: {int(exclusion_mask.sum()):,}')
    print(f'Local incidents suppressed: {int(suppressed_any.sum()):,}')
    return classified


def create_filtered_headlines_table(classified):
    cols = [
        'date', 'headline_text', 'primary_category', 'categories',
        'is_market_prices', 'is_supply_geopolitics', 'is_consumer_fuel',
        'market_match_rule', 'supply_match_rule', 'consumer_match_rule',
    ]
    return (
        classified.loc[classified['is_oil_relevant'], cols]
        .sort_values(['date', 'primary_category', 'headline_text'])
        .reset_index(drop=True)
    )


def create_daily_oil_table(filtered, daily_counts_file):
    print('\nCreating daily oil-news measures...')
    daily = pd.read_csv(daily_counts_file, parse_dates=['date'])

    total = filtered.groupby('date').size().rename('oil_headlines')
    market = filtered.groupby('date')['is_market_prices'].sum().rename('market_price_headlines')
    supply = filtered.groupby('date')['is_supply_geopolitics'].sum().rename('supply_geopolitics_headlines')
    consumer = filtered.groupby('date')['is_consumer_fuel'].sum().rename('consumer_fuel_headlines')

    counts = pd.concat([total, market, supply, consumer], axis=1).reset_index()
    daily = daily.merge(counts, on='date', how='left', validate='one_to_one')

    count_columns = [
        'oil_headlines', 'market_price_headlines',
        'supply_geopolitics_headlines', 'consumer_fuel_headlines',
    ]
    for col in count_columns:
        daily[col] = daily[col].fillna(0).astype(int)

    mapping = {
        'oil': 'oil_headlines',
        'market_price': 'market_price_headlines',
        'supply_geopolitics': 'supply_geopolitics_headlines',
        'consumer_fuel': 'consumer_fuel_headlines',
    }
    for prefix, count_col in mapping.items():
        daily[f'{prefix}_share'] = daily[count_col] / daily['total_headlines']
        daily[f'{prefix}_share_per_1000'] = daily[f'{prefix}_share'] * 1000

    daily = daily.sort_values('date').reset_index(drop=True)
    print(f'Dates with at least one oil headline: {(daily["oil_headlines"] > 0).sum():,}')
    return daily


def merge_with_brent(daily_oil, abc_brent_file):
    print('\nMerging categorized oil news with Brent prices...')
    abc_brent = pd.read_csv(abc_brent_file, parse_dates=['date'])
    merged = abc_brent.merge(
        daily_oil.drop(columns=['total_headlines']),
        on='date', how='left', validate='one_to_one'
    )
    count_columns = [
        'oil_headlines', 'market_price_headlines',
        'supply_geopolitics_headlines', 'consumer_fuel_headlines',
    ]
    for col in count_columns:
        merged[col] = merged[col].fillna(0).astype(int)
    share_columns = [c for c in merged.columns if c.endswith('_share') or c.endswith('_per_1000')]
    for col in share_columns:
        merged[col] = merged[col].fillna(0.0)
    merged = merged.sort_values('date').reset_index(drop=True)
    print(f'Final matched trading dates: {len(merged):,}')
    return merged


def create_review_samples(classified, filtered):
    matched_parts = []
    for category in ['market_prices', 'supply_geopolitics', 'consumer_fuel']:
        rows = filtered[filtered['primary_category'] == category]
        if len(rows):
            matched_parts.append(rows.sample(n=min(50, len(rows)), random_state=42))
    matched = pd.concat(matched_parts, ignore_index=True).sort_values(['primary_category', 'date'])

    broad = classified['normalized_headline'].str.contains(BROAD_CANDIDATE_PATTERN, regex=True, na=False)
    rejected = classified.loc[
        broad & ~classified['is_oil_relevant'],
        ['date', 'headline_text', 'is_excluded', 'exclusion_reason', 'suppression_reason']
    ].copy()
    if len(rejected):
        rejected = rejected.sample(n=min(150, len(rejected)), random_state=42).sort_values('date')

    multi = filtered[filtered['categories'].str.contains(';', regex=False, na=False)]
    if len(multi):
        multi = multi.sample(n=min(100, len(multi)), random_state=42).sort_values('date')
    return matched, rejected, multi


def print_summary(filtered, daily):
    print('\n' + '=' * 60)
    print('FILTER SUMMARY')
    print('=' * 60)
    print(f'Filtered date range: {filtered["date"].min().date()} to {filtered["date"].max().date()}')
    print('\nPrimary-category counts:')
    print(filtered['primary_category'].value_counts().to_string())
    print(f'\nHeadlines matching more than one category: {filtered["categories"].str.contains(";", regex=False, na=False).sum():,}')
    print('\nTop 10 dates by total oil coverage:')
    cols = ['date', 'total_headlines', 'oil_headlines', 'market_price_headlines', 'supply_geopolitics_headlines', 'consumer_fuel_headlines', 'oil_share_per_1000']
    print(daily.sort_values('oil_headlines', ascending=False).head(10)[cols].to_string(index=False))


def save_results(filtered, daily, merged, matched, rejected, multi):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(FILTERED_HEADLINES_OUTPUT, index=False, date_format='%Y-%m-%d')
    daily.to_csv(OIL_DAILY_OUTPUT, index=False, date_format='%Y-%m-%d')
    merged.to_csv(FINAL_MERGED_OUTPUT, index=False, date_format='%Y-%m-%d')
    matched.to_csv(MATCHED_SAMPLE_OUTPUT, index=False, date_format='%Y-%m-%d')
    rejected.to_csv(REJECTED_SAMPLE_OUTPUT, index=False, date_format='%Y-%m-%d')
    multi.to_csv(MULTI_CATEGORY_SAMPLE_OUTPUT, index=False, date_format='%Y-%m-%d')
    print('\nSaved files:')
    for path in [FILTERED_HEADLINES_OUTPUT, OIL_DAILY_OUTPUT, FINAL_MERGED_OUTPUT,
                 MATCHED_SAMPLE_OUTPUT, REJECTED_SAMPLE_OUTPUT, MULTI_CATEGORY_SAMPLE_OUTPUT]:
        print(path)


def main():
    print('=' * 60)
    print('GeoOil-Pulse: Final Oil Headline Filtering')
    print('=' * 60)
    for file_path in [ABC_RAW_FILE, ABC_DAILY_COUNTS_FILE, ABC_BRENT_DAILY_FILE]:
        check_file_exists(file_path)
    news = load_and_clean_abc_headlines(ABC_RAW_FILE)
    classified = classify_oil_headlines(news)
    filtered = create_filtered_headlines_table(classified)
    daily = create_daily_oil_table(filtered, ABC_DAILY_COUNTS_FILE)
    merged = merge_with_brent(daily, ABC_BRENT_DAILY_FILE)
    matched, rejected, multi = create_review_samples(classified, filtered)
    print_summary(filtered, daily)
    save_results(filtered, daily, merged, matched, rejected, multi)
    print('\nFinal oil-headline filtering completed successfully.')


if __name__ == '__main__':
    main()
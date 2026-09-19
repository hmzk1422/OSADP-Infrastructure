"""Final validation over all 48 rebuilt matched files (2026-02-01 -> 2026-02-02)."""
import sys
from pathlib import Path

import polars as pl

MATCHED = Path(r'c:\Users\Himaska\Documents\Projects\Business_Projects\intraday-momentum-composite\research\data_v2\matched\BTCUSDT')

files = sorted(MATCHED.glob('*.parquet'))
print(f'files: {len(files)}')

tot = pl.DataFrame()
agg = {
    'rows': 0,
    'null_best_bid': 0,
    'null_best_ask': 0,
    'crossed': 0,
    'bid_order_viol': 0,
    'ask_order_viol': 0,
    'lvl20_null_bid': 0,
    'lvl20_null_ask': 0,
    'bd_null': 0,
    'bd_rows': 0,
}
spreads_min, spreads_max = 1e18, -1e18

for f in files:
    lf = pl.scan_parquet(str(f))
    s = lf.select(
        pl.len().alias('rows'),
        pl.col('best_bid_price').is_null().sum().alias('null_best_bid'),
        pl.col('best_ask_price').is_null().sum().alias('null_best_ask'),
        (pl.col('best_bid_price') >= pl.col('best_ask_price')).sum().alias('crossed'),
        (pl.col('best_bid_price') < pl.col('bid_price_2')).sum().alias('bid_order_viol'),
        (pl.col('best_ask_price') > pl.col('ask_price_2')).sum().alias('ask_order_viol'),
        pl.col('bid_price_20').is_null().sum().alias('lvl20_null_bid'),
        pl.col('ask_price_20').is_null().sum().alias('lvl20_null_ask'),
        pl.col('bd_depth_m1').is_null().sum().alias('bd_null'),
        (pl.col('best_ask_price') - pl.col('best_bid_price')).min().alias('sp_min'),
        (pl.col('best_ask_price') - pl.col('best_bid_price')).max().alias('sp_max'),
        pl.col('bd_depth_m1').is_not_null().sum().alias('bd_rows'),
        pl.col('best_bid_price').is_not_null().count().alias('tob_rows'),
    ).collect().row(0, named=True)
    for k in ('rows', 'null_best_bid', 'null_best_ask', 'crossed', 'bid_order_viol',
              'ask_order_viol', 'lvl20_null_bid', 'lvl20_null_ask', 'bd_null', 'bd_rows'):
        agg[k] += s[k]
    spreads_min = min(spreads_min, s['sp_min'] if s['sp_min'] is not None else 1e18)
    spreads_max = max(spreads_max, s['sp_max'] if s['sp_max'] is not None else -1e18)

print(agg)
print(f'spread min across all files: {spreads_min}')
print(f'spread max across all files: {spreads_max}')
print(f'bd coverage: {agg["bd_rows"]}/{agg["rows"]} = {agg["bd_rows"]/agg["rows"]:.3%}')
print(f'tob coverage: {agg["rows"]-agg["null_best_bid"]}/{agg["rows"]}')

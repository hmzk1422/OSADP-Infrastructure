"""
Run only the native Rust GP stage on configured L2 data.

USE THIS ONLY FOR TESTING GP only.
"""


from __future__ import annotations

from strategy_system.exchange.loader import load_L2
from strategy_system.strategy.moe.ops.settings import load_settings
from strategy_system.strategy.moe.strategy import run_gp_new


def main() -> None:
    settings = load_settings()
    data = load_L2(**settings["data"].get("orderbook", {}))
    result = run_gp_new(data, settings)
    print(f"rows={len(result.features)} terminals={result.terminals.shape[1]}")
    print(f"selected_features={len(result.feature_set.names)}")
    print(f"search_stats={result.feature_set.stats}")
    for feature in result.feature_set.library["features"]:
        print(f'{feature["name"]} = {feature["formula"]}')


if __name__ == "__main__":
    main()

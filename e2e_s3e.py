"""Temporary end-to-end S3E run (deleted after validation)."""
from strategy_system.strategy.moe.strategy import load_data, load_settings, strategy

settings = load_settings()  # defaults: file_limit 10

raw = load_data(settings)
print(f"raw rows: {len(raw)} cols: {len(raw.columns)}")

result = strategy(raw, settings)
print(f"features rows: {len(result.features)}")
print(f"regimes: {result.regimes.shape if result.regimes is not None else None}")
print(f"events: {len(result.events)}")
print(f"gp model: {'fitted' if result.gp is not None else 'none'}")
print(f"metalabel: {'fitted' if result.metalabel is not None else 'none'}")
print("--- report ---")
print(result.report)
print("E2E OK")

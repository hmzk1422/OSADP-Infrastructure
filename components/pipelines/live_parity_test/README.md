# Live strategy parity runner

This Cargo binary accepts arbitrary Python and Rust executables. It passes
`PARITY_DATA_PATH`, `PARITY_SYMBOL`, `PARITY_TAG`, `PARITY_PYTHON_OUT`, and
`PARITY_RUST_OUT` to both child processes. Each process must write its result
CSV with a numeric timestamp column and the requested numeric feature/trade
columns. Repeat `--python-env KEY=VALUE` or `--rust-env KEY=VALUE` to set
child-specific environment variables without adding strategy-specific options
to the runner.

The runner aligns rows by timestamp, permits only extra leading Rust warm-up
rows, checks cosine similarity and MAE for feature columns, and requires exact
equality for trade-path columns.

Example for OFMR on one day of matched BTCUSDT data:

```powershell
cargo run --manifest-path components\pipelines\live_parity_test\Cargo.toml -- `
  --data-path market_data\matched `
  --symbol BTCUSDT `
  --tag 2025-06-30 `
  --python-program python `
  --python-arg strategy_system\strategy\ofmr\test_strategy_parity.py `
  --rust-env OFMR_CONFIDENCE_THRESHOLD=0.2 `
  --rust-program cargo `
  --rust-arg run `
  --rust-arg --manifest-path `
  --rust-arg components\pipelines\uets_trades\Cargo.toml `
  --rust-arg --bin `
  --rust-arg playback_ofmr `
  --rust-arg -- `
  --rust-arg --model `
  --rust-arg uets\models\ofmr\trader.txt `
  --feature-cols Column_0,Column_1,Column_2,Column_3,Column_4,Column_5,Column_6,Column_7,Column_8,Column_9,Column_10,Column_11,Column_12,Column_13,Column_14 `
  --trade-cols decision,position
```

`--feature-cols` and `--trade-cols` describe the output schema; they are not
fixed by the runner. Output CSVs are written under
`build\live_parity_test\` by default.

## Separate research-artifact parity

The Cargo runner compares the causal Python and Rust live paths. It does not
compare either live path with the study's label-aware walk-forward trade log.
To check production feature ordering and Python model scores against the saved
study state space and fold models, run this from the repository root after
`ml_state_optimized.py` has generated its artifacts:

```powershell
.\venv\Scripts\python.exe strategy_system\strategy\ofmr\test_ml_state_optimized_parity.py
```

By default, the study-artifact test reads
`strategy_system\strategy\ofmr\models\state_space_btcusdt_720files.parquet`
and `build\walkforward_models\`. Pass `--state-space`, `--catalog`, or
`--model-dir` to use artifacts from another study run. The two tests answer
different questions: study features/scores and threshold signals are checked
against the study's out-of-sample folds; live trade paths are compared
separately between Python and Rust.

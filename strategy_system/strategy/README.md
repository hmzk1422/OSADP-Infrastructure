# Strategy System
Last updated since v1.0.0

Framework responsible for resesarching quantitative edges and putting them into the algorithm. Understand the following structure.
```bash
strategy_system
|_ data         # Select between live data and synthetic data
|_ tests        # includes tests like monte carlo, purged chronological wf, etc
|_ strategy     # Code your strategies here
```
## Strategy
Structure of strategy folder
```bash
strategy
|_ Name_of_strategy
    |_ research.pdf         # provide research paper to prove your edge
    |_ README.md            # journal changes to your strategy
    |_ compute_features.rs  # placed into uets/executor/ml_feature
    |_ state_builder.rs     # placed into uets/executor/strategy
    |_ main.rs              # calls the function, feeds it data
    |_ ...
```
NOTE:
1. Two ways for this. Your strategy folder must result in ONNX model, or your strategy's code, computation must be coded using rust.
    - ONNX allows you to use python whilst maintaining dynamic, versatility of UETS system. However it is noticeably slower due to I/O overhead cost.
    - Rust hardcoding guarantees integrity of your strategy and is very fast. For live-production, this is prefered. However, use python to keep research simple and fast, or if it may involve using unsupported ml algos.
2. If you choose to hardcode (preferably use this structure even with python)
    - compute_feature.rs --> purely for computing features
    - state_builder.rs --> rules of strategy, building data (ex: triple barrier)
    - research.pdf --> to document mathematically how the edge works.

## Notes
[PAUSED] Mixture of Experts 
[PAUSED] Volatility Decay Capture
[APPROVED] Orderflow Mean Reversion
[] Statistical Arbitrage
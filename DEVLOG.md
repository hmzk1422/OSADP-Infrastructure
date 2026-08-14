# Developer Log
Documenting architectural planning and engineering hurdles.

# Unified Backtest/Live Architecture System
Reference: feature/unified-architectural-system
```
          +---------------------------------+
          |        Trading Strategy         |
          |       (Agnostic Core Logic)     |
          +---------------------------------+
                           │ model.txt
                           ▼
          +---------------------------------+
          |         Execution Layer         |
          |     (Shared Engine / Core)      |
          +---------------------------------+
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    +---------------+             +---------------+
    | Histbacktest  |             |Live Execution |
    | (Parquet/CSV) |             | (WebSockets)  |
    +---------------+             +---------------+
```
Two systems, one strategy. Using shared architecture prevents inconsistency between backtest and live executions.
- **Trading Strategy**: Purely for defining the strategy code. 
    - ml_state_features.py: Computing the features, completely separate to ensure with very high confidence no bleeding or lookahead bias happens. 
    - ml_state_optimized.py: Composing features together with market such as calculating triple barrier or model training.
- **Execution Layer**: ml_state_runtime.py connected with ml_state_optimized as abstraction for true execution, querying data, providing commands, etc. Connects to txt file obtained from trading strategy.
- **Histbacktest/Live Execution**: Execution layer connects with histbackteest and live execution to query for data. 

Note: Should latency matter, migrate execution layer and histbacktest/live execution into rust.
# Hurdles of Historical Backtest Performance and Data Management
Reference: fix/historical-backtest
- Old historical backtest was a dual architectural system that ingested csv files that relied on querying through live websockets which introduced two problems. Model must be specialized to one asset and querying data is very slow. As such, download shared aggregated data from historical archives in parquet format.
- Due to the large file size of processing months long orderflow/websocket data, loading to main ram introduces performance issues relating to Out-Of-Memory (OOM) crashes. In considering hardware limitations, this was the solution.
    - Chunking parquet files to smaller segments to reduce load in main memory. However, certain known as recursive indicators requires continuous stream of data and chunking resets it. Even with SessionFeatureContext class set, test_chunking_loss showed a major +-5-10% error due to compounding error. Example of recursive feature, VPIN. Temporarily, recursive feature columns are selected out to process without chunking which may introduce peroformance issues.
    - Migration from Pandas to Numpy and Polars helps significantly improve performance where numpy helps process calculations for polars that are not included in pandas without slowing down process. Polar's multithreading and lazy scanning feature helps run algorithm under certain load.
    - Virtual Memory and Disk allocation. With polars, partition tasks to drive to reduce as much load in the main memory as possible at the cost of possible disk thrashing. However, when considering limited hardware resources, this method is advantageous.
    - Biggest process bottlenecks lies within sorting algorithm and model training, where sorting takes up O(nlog(n)) complexity which could be easily avoided provided endpoints are already sorted. Tree model training with O(n*m), where m represent features, is the main priority.
- Why is ml_state_optimized to ml_state_optimized_csv transition considered lossy? All features are recursive and affected slightly by chunking. For example, micro-cvd has a 1.2*10^-5 error margin from mean value 17. Very small error when compounded over time, becomes an error that could affect ML decision.
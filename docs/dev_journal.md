# Integrity of strataegy from python to rust — 23/08/2026
Moving from python to rust is problematic where its hard to keep the logic and parameters consistent in such a large system. Strangely, strategy consistently returned a consistently negative return even with zero commissions. 
- Solved by running long-short system. The same strategy, long and the same strategy short simultaneously. If it returned approximately symmetric inverse returns of one another with no deviation, then the strategy is preserved.
- Execute the UETS engine with txt file directly on walk forward data used in research, then switch research file with txt file. Run correlation/MAE test over a series of execution to ensure strategy integrity is maintained. While MAE returned moderate deviation (average 0.19rr +- 0.08rr), the strong positive correlation indicates that the Rust implementation preserves the directional structure of the research implementation, while the moderate MAE suggests remaining implementation differences requiring investigation.

# Production-Grade Failure Modes — 06/09/2026

Although the core algorithm is largely complete, recent live-paper testing exposed several production-critical failure modes that must be resolved before live deployment.

### 1. Global Time Synchronization

As the system expanded to multiple strategies, venues, dashboards and research metrics, the laptop's wall clock became an unreliable reference. A local clock error (including CMOS/time-sync issues) can distort cross-process event timestamps and make execution measurements appear accurate while only reflecting local elapsed time.

The system therefore uses a synchronized global time reference for event correlation, live dashboard measurement and cross-strategy comparison, while monotonic clocks remain responsible for local durations such as RTT, order latency and processing time.

This is also important when comparing strategies on the same asset: RSI and other interval-based features may legitimately differ because each strategy can observe the market at a different point in time. A common time reference allows those states to be aligned and compared fairly.

Exchange timestamps remain useful for venue-specific data, but relying on repeated exchange API calls as the application's primary clock would limit live observability and introduce unnecessary venue dependencies.

### 2. Network Failure / Fail-Safe Execution

A prolonged network outage during an active trigger regime exposed a critical failure: the strategy did not reliably detect stale connectivity and continued operating on invalid external state.

When connectivity returned, accumulated actions were executed together, resulting in hundreds of ETH trades in the paper environment. With real capital, this could have produced catastrophic unintended exposure.

**Failure chain:** network loss → stale state → strategy continues → recovery → execution burst.

The system therefore needs explicit connectivity/staleness states and a hard fail-safe that prevents trading until market-data and execution state are safely re-established.

A separate concern is **cross-venue resource contention**. Isolated asset processes limit blast radius, but excessive reconnects/traffic from one failing venue could consume shared network resources and degrade otherwise healthy venues. This remains under investigation.


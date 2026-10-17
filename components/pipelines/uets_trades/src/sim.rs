//! Shared simulated execution — one implementation of fill mechanics,
//! slippage, fees, positions and barriers for **both** `live_test` (paper)
//! and `playback` (replay), so simulated results are identical across modes.
//!
//! Accounting lives in [`Book`] (also used by `live_trade`), which means
//! portfolio math — cash, equity, exposure, realized/unrealized PnL, wins,
//! losses and R-multiples — is the same code in every execution mode.

use crate::backend::BackendError;
use crate::triple_barrier::TripleBarrier;
use crate::types::{
    Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, OrderSide, OrderStatus,
    OrderType,
};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

/// Tolerance for position "flat" comparisons.
const EPS: f64 = 1e-12;

/// Execution assumptions for simulated fills.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Half-spread applied at execution, in bps.
    pub spread_bps: f64,
    /// Adverse slippage applied at execution, in bps.
    pub slippage_bps: f64,
    /// Taker fee per fill notional, in bps.
    pub fee_bps: f64,
    /// Capital allocated lazily to strategies with no explicit allocation.
    pub starting_capital: f64,
    /// When set, a market order's *first* fill covers this fraction of the
    /// quantity (partial fill); the remainder fills on the next market
    /// update. `None` = fill fully in one print.
    pub partial_fill_ratio: Option<f64>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            spread_bps: 1.0,
            slippage_bps: 1.0,
            fee_bps: 4.0,
            starting_capital: 10_000.0,
            partial_fill_ratio: None,
        }
    }
}

/// Position key: strategy instance + symbol (tuple identity, never parsed).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PosKey {
    strategy_id: String,
    symbol: String,
}

/// An open (or partially closed) position with its entry risk.
#[derive(Clone, Debug)]
struct Position {
    /// Signed quantity: positive = long, negative = short.
    qty: f64,
    /// Quantity-weighted average entry price.
    avg_entry: f64,
    /// Entry fees still attached to the open portion.
    entry_fee: f64,
    /// Net realised pnl accumulated while scaling out (settled at flat).
    realized_bucket: f64,
    /// Total initial risk in quote units (from the entry triple barrier).
    risk_total: Option<f64>,
    /// Entry barrier, monitored while the position is open.
    barrier: Option<TripleBarrier>,
}

/// Portfolio accounting shared by all execution modes.
#[derive(Clone, Debug, Default)]
pub struct Book {
    capital: BTreeMap<String, Capital>,
    positions: BTreeMap<PosKey, Position>,
    marks: BTreeMap<String, f64>,
}

impl Book {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate capital for a strategy instance.
    pub fn with_capital(mut self, strategy_id: impl Into<String>, starting: f64) -> Self {
        self.capital
            .insert(strategy_id.into(), Capital::new(starting));
        self
    }

    /// Ensure a strategy has capital (lazy allocation at `starting`).
    pub fn ensure_capital(&mut self, strategy_id: &str, starting: f64) {
        self.capital
            .entry(strategy_id.to_string())
            .or_insert_with(|| Capital::new(starting));
    }

    /// Replace a strategy's allocation (used by explicit `with_capital`
    /// constructors).
    pub fn set_capital(&mut self, strategy_id: impl Into<String>, starting: f64) {
        self.capital
            .insert(strategy_id.into(), Capital::new(starting));
    }

    /// Latest capital snapshot for a strategy instance.
    pub fn capital(&self, strategy_id: &str) -> Option<Capital> {
        self.capital.get(strategy_id).cloned()
    }

    /// Merge the latest marks.
    pub fn set_marks(&mut self, marks: &BTreeMap<String, f64>) {
        for (symbol, price) in marks {
            self.marks.insert(symbol.clone(), *price);
        }
    }

    /// Latest known mark for a symbol.
    pub fn mark(&self, symbol: &str) -> Option<f64> {
        self.marks.get(symbol).copied()
    }

    /// Open quantity for (strategy, symbol), signed.
    pub fn position_qty(&self, strategy_id: &str, symbol: &str) -> Option<f64> {
        self.positions
            .get(&PosKey {
                strategy_id: strategy_id.to_string(),
                symbol: symbol.to_string(),
            })
            .map(|p| p.qty)
    }

    /// How a fill splits against the existing position: the quantity that
    /// closes the current position and the quantity that opens (or adds).
    fn split_fill(existing_qty: f64, dir: f64, qty: f64) -> (f64, f64) {
        let closing = if existing_qty != 0.0 && existing_qty.signum() != dir {
            existing_qty.abs().min(qty)
        } else {
            0.0
        };
        (closing, qty - closing)
    }

    /// Why a *buy* of `qty` at `price` (opening portion) cannot execute, or
    /// `None` if it can. Sells always execute (they free cash or collect
    /// proceeds).
    pub fn capacity_error(
        &self,
        strategy_id: &str,
        side: OrderSide,
        qty: f64,
        price: f64,
        fee: f64,
    ) -> Option<String> {
        if side == OrderSide::Sell {
            return None;
        }
        let capital = self.capital.get(strategy_id)?;
        let required = price * qty + fee;
        if capital.cash + EPS < required {
            Some(format!(
                "insufficient cash: required {required:.4}, available {:.4}",
                capital.cash
            ))
        } else {
            None
        }
    }

    /// Apply one fill: updates cash, position, realised/unrealized PnL,
    /// fees, exposure, win/loss counts and R-multiples.
    ///
    /// `entry_barrier` is attached when this fill opens (or adds to) a
    /// position; it defines the initial risk used for R-multiples.
    pub fn apply_fill(
        &mut self,
        strategy_id: &str,
        symbol: &str,
        side: OrderSide,
        qty: f64,
        price: f64,
        fee: f64,
        entry_barrier: Option<&TripleBarrier>,
    ) -> Result<(), String> {
        if !(qty.is_finite() && qty > 0.0 && price.is_finite() && price > 0.0) {
            return Err(format!("invalid fill: qty={qty} price={price}"));
        }
        self.ensure_capital(strategy_id, 0.0);
        let dir = match side {
            OrderSide::Buy => 1.0,
            OrderSide::Sell => -1.0,
        };
        let key = PosKey {
            strategy_id: strategy_id.to_string(),
            symbol: symbol.to_string(),
        };

        let existing_qty = self.positions.get(&key).map(|p| p.qty).unwrap_or(0.0);
        let (closing, opening) = Self::split_fill(existing_qty, dir, qty);

        // ---- Closing portion: realise PnL, settle fees, count trades ----
        if closing > 0.0 {
            let exit_fee = fee * (closing / qty);
            let entry_fee_share = {
                let p = self.positions.get(&key).unwrap();
                let open_abs = p.qty.abs();
                if open_abs > 0.0 {
                    p.entry_fee * (closing / open_abs)
                } else {
                    0.0
                }
            };
            let (net, became_flat, trade_stats) = {
                let p = self.positions.get_mut(&key).unwrap();
                let gross = (price - p.avg_entry) * closing * p.qty.signum();
                let net = gross - entry_fee_share - exit_fee;
                p.realized_bucket += net;
                p.entry_fee -= entry_fee_share;
                p.qty -= closing * p.qty.signum();
                let flat = p.qty.abs() < EPS;
                if flat {
                    p.qty = 0.0;
                }
                let stats = if flat {
                    Some((p.realized_bucket, p.risk_total.filter(|r| *r > 0.0)))
                } else {
                    None
                };
                (net, flat, stats)
            };

            // Cash movement for the closing portion (sell = +, buy-back = -).
            let cash_delta = if dir > 0.0 {
                price * closing - exit_fee
            } else {
                -(price * closing + exit_fee)
            };
            let capital = self.capital.get_mut(strategy_id).unwrap();
            capital.cash += cash_delta;
            capital.realized_pnl += net;
            capital.fees_paid += exit_fee;

            if became_flat {
                if let Some((bucket, risk)) = trade_stats {
                    let capital = self.capital.get_mut(strategy_id).unwrap();
                    capital.trade_count += 1;
                    if bucket > 0.0 {
                        capital.win_count += 1;
                    } else if bucket < 0.0 {
                        capital.loss_count += 1;
                    }
                    if let Some(risk) = risk {
                        capital.r_multiple_sum += bucket / risk;
                        capital.r_trades += 1;
                    }
                }
                self.positions.remove(&key);
            }
        }
        // ---- Opening portion: consume cash, upsert position ----
        if opening > 0.0 {
            let open_fee = fee * (opening / qty);
            let capital = self.capital.get_mut(strategy_id).unwrap();
            if dir > 0.0 && capital.cash + EPS < price * opening + open_fee {
                return Err(format!(
                    "insufficient cash: required {}, available {:.4}",
                    price * opening + open_fee,
                    capital.cash
                ));
            }
            let cash_delta = if dir > 0.0 {
                -(price * opening + open_fee)
            } else {
                price * opening - open_fee
            };
            capital.cash += cash_delta;
            capital.fees_paid += open_fee;

            let risk_total = entry_barrier.map(|b| b.risk_per_unit() * opening);
            let entry = self.positions.entry(key).or_insert_with(|| Position {
                qty: 0.0,
                avg_entry: price,
                entry_fee: 0.0,
                realized_bucket: 0.0,
                risk_total,
                barrier: entry_barrier.cloned(),
            });
            if entry.qty.abs() < EPS {
                entry.qty = dir * opening;
                entry.avg_entry = price;
                entry.entry_fee = open_fee;
                entry.realized_bucket = 0.0;
                entry.risk_total = risk_total;
                entry.barrier = entry_barrier.cloned();
            } else {
                // Adding to an existing position: weighted average entry.
                let total = entry.qty.abs() + opening;
                entry.avg_entry = (entry.avg_entry * entry.qty.abs() + price * opening) / total;
                entry.qty += dir * opening;
                entry.entry_fee += open_fee;
                // Keep the first barrier: its risk defines the trade.
            }
        }

        Ok(())
    }

    /// Revalue every strategy: equity, unrealized, exposure and margin.
    pub fn mark_to_market(&mut self) {
        let mut per: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
        for (key, pos) in &self.positions {
            let mark = self
                .marks
                .get(&key.symbol)
                .copied()
                .unwrap_or(pos.avg_entry);
            let value = pos.qty * mark;
            let unreal = (mark - pos.avg_entry) * pos.qty;
            let slot = per
                .entry(key.strategy_id.clone())
                .or_insert((0.0, 0.0, 0.0));
            slot.0 += unreal;
            slot.1 += value.abs();
            slot.2 += value;
        }
        for (strategy_id, capital) in self.capital.iter_mut() {
            let (unreal, gross, net) = per.get(strategy_id).cloned().unwrap_or((0.0, 0.0, 0.0));
            capital.unrealized_pnl = unreal;
            capital.equity = capital.cash + net;
            capital.gross_exposure = gross;
            capital.net_exposure = net;
            capital.margin_used = gross;
            capital.available_margin = (capital.equity - gross).max(0.0);
            capital.effective_leverage = if capital.equity > 0.0 {
                gross / capital.equity
            } else {
                0.0
            };
        }
    }

    /// Open positions that carry a barrier (for exit monitoring):
    /// (strategy, symbol, qty, is_long, barrier).
    pub(crate) fn barrier_positions(&self) -> Vec<(String, String, f64, bool, TripleBarrier)> {
        self.positions
            .iter()
            .filter_map(|(key, pos)| {
                pos.barrier.as_ref().map(|b| {
                    (
                        key.strategy_id.clone(),
                        key.symbol.clone(),
                        pos.qty,
                        b.is_long,
                        b.clone(),
                    )
                })
            })
            .collect()
    }
}

/// Fill outcome for one fill attempt.
enum FillOutcome {
    Idle,
    Waiting,
    Filled,
    Rejected(String),
}

/// The simulated execution engine shared by paper and playback backends.
///
/// Lifecycle: `OrderAccepted` at submission; fills happen immediately when a
/// mark is known, otherwise on the next market update. Barrier exits are
/// synthesised as internal exit orders.
pub struct SimEngine {
    pub config: SimConfig,
    pub book: Book,
    orders: BTreeMap<OrderId, Order>,
    events: Vec<ExecutionEvent>,
    barrier_seq: u64,
}

impl SimEngine {
    pub fn new(config: SimConfig) -> Self {
        Self {
            config,
            book: Book::new(),
            orders: BTreeMap::new(),
            events: Vec::new(),
            barrier_seq: 0,
        }
    }

    /// Engine with a pre-allocated strategy capital.
    pub fn with_capital(config: SimConfig, strategy_id: impl Into<String>, starting: f64) -> Self {
        let mut engine = Self::new(config);
        engine.book.set_capital(strategy_id, starting);
        engine
    }

    /// Accept an order: validates, records it, and fills immediately if a
    /// mark for the symbol is already known.
    pub fn submit(
        &mut self,
        request: OrderRequest,
        now: DateTime<Utc>,
    ) -> Result<OrderId, BackendError> {
        if !(request.qty.is_finite() && request.qty > 0.0) {
            let reason = format!("invalid quantity: {}", request.qty);
            self.events.push(ExecutionEvent::OrderRejected {
                order_id: request.id.clone(),
                reason: reason.clone(),
                at: now,
            });
            return Err(BackendError::OrderRejected(reason));
        }
        if self.orders.contains_key(&request.id) {
            return Err(BackendError::DuplicateOrderId(request.id.0.clone()));
        }
        self.book
            .ensure_capital(&request.strategy_id, self.config.starting_capital);

        let id = request.id.clone();
        let symbol = request.symbol.clone();
        let mut order = Order::new(id.clone(), request, now);
        order.status = OrderStatus::Accepted;
        order.updated_at = now;
        self.orders.insert(id.clone(), order);
        self.events.push(ExecutionEvent::OrderAccepted {
            order_id: id.clone(),
            symbol,
            at: now,
        });

        // Fill immediately when a mark is known; otherwise wait for the
        // next market update.
        match self.try_fill(&id, now) {
            FillOutcome::Rejected(reason) => Err(BackendError::OrderRejected(reason)),
            _ => Ok(id),
        }
    }

    /// Cancel a live order. Terminal orders cannot be cancelled.
    pub fn cancel(&mut self, order_id: &OrderId, now: DateTime<Utc>) -> Result<(), BackendError> {
        let order = self
            .orders
            .get_mut(order_id)
            .ok_or_else(|| BackendError::Internal(format!("unknown order {order_id}")))?;
        if order.status.is_terminal() {
            return Err(BackendError::Internal(format!(
                "order {order_id} already {:?}",
                order.status
            )));
        }
        order.status = OrderStatus::Cancelled;
        order.updated_at = now;
        self.events.push(ExecutionEvent::OrderCancelled {
            order_id: order_id.clone(),
            at: now,
        });
        Ok(())
    }

    /// Attempt to fill one live order at the current mark.
    fn try_fill(&mut self, order_id: &OrderId, now: DateTime<Utc>) -> FillOutcome {
        let Some(order) = self.orders.get(order_id) else {
            return FillOutcome::Idle;
        };
        if !matches!(
            order.status,
            OrderStatus::Accepted | OrderStatus::PartiallyFilled
        ) {
            return FillOutcome::Idle;
        }
        let request = order.request.clone();
        let filled_so_far = order.filled_qty;
        let remaining = (request.qty - filled_so_far).max(0.0);
        if remaining < EPS {
            return FillOutcome::Idle;
        }
        let Some(mark) = self.book.mark(&request.symbol) else {
            return FillOutcome::Waiting;
        };

        // Decide fill quantity and execution price.
        let (fill_qty, price) = match request.order_type {
            OrderType::Market => {
                let mut q = remaining;
                if filled_so_far == 0.0 {
                    if let Some(ratio) = self.config.partial_fill_ratio {
                        q = (remaining * ratio).clamp(EPS, remaining);
                    }
                }
                let edge = (self.config.spread_bps / 2.0 + self.config.slippage_bps) / 10_000.0;
                let px = match request.side {
                    OrderSide::Buy => mark * (1.0 + edge),
                    OrderSide::Sell => mark * (1.0 - edge),
                };
                (q, px)
            }
            OrderType::Limit => {
                let Some(limit) = request.price else {
                    let reason = "limit order without a price".to_string();
                    self.reject_order(order_id, reason.clone(), now);
                    return FillOutcome::Rejected(reason);
                };
                let cross = match request.side {
                    OrderSide::Buy => mark <= limit,
                    OrderSide::Sell => mark >= limit,
                };
                if !cross {
                    return FillOutcome::Waiting;
                }
                (remaining, limit)
            }
        };

        let fee = self.config.fee_bps / 10_000.0 * price * fill_qty;
        let apply = self.book.apply_fill(
            &request.strategy_id,
            &request.symbol,
            request.side,
            fill_qty,
            price,
            fee,
            request.barrier.as_ref(),
        );
        if let Err(reason) = apply {
            self.reject_order(order_id, reason.clone(), now);
            return FillOutcome::Rejected(reason);
        }

        // Update order state and emit lifecycle events.
        let fully_filled = {
            let order = self.orders.get_mut(order_id).unwrap();
            order.record_fill(fill_qty, price, fee, now);
            order.filled_qty + EPS >= request.qty
        };
        self.book.mark_to_market();
        if fully_filled {
            self.events.push(ExecutionEvent::OrderFilled {
                order_id: order_id.clone(),
                filled_qty: fill_qty,
                fill_price: price,
                fee,
                at: now,
            });
        } else {
            self.events.push(ExecutionEvent::OrderPartiallyFilled {
                order_id: order_id.clone(),
                filled_qty: fill_qty,
                fill_price: price,
                fee,
                at: now,
            });
        }
        self.events.push(ExecutionEvent::Fill(FillEvent {
            order_id: order_id.clone(),
            symbol: request.symbol.clone(),
            side: request.side,
            qty: fill_qty,
            price,
            fee,
            executed_at: now,
        }));

        if fully_filled {
            FillOutcome::Filled
        } else {
            FillOutcome::Filled
        }
    }

    /// Mark an order rejected (fill-time validation failure).
    fn reject_order(&mut self, order_id: &OrderId, reason: String, now: DateTime<Utc>) {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.status = OrderStatus::Rejected;
            order.updated_at = now;
        }
        self.events.push(ExecutionEvent::OrderRejected {
            order_id: order_id.clone(),
            reason,
            at: now,
        });
    }

    /// Mark-to-market update: merge marks, fill live orders, monitor
    /// triple barriers, and revalue every strategy.
    pub fn on_market_update(&mut self, marks: &BTreeMap<String, f64>, now: DateTime<Utc>) {
        self.book.set_marks(marks);

        let live_ids: Vec<OrderId> = self
            .orders
            .iter()
            .filter(|(_, o)| {
                matches!(
                    o.status,
                    OrderStatus::Accepted | OrderStatus::PartiallyFilled
                )
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in live_ids {
            self.try_fill(&id, now);
        }

        self.check_barriers(now);
        self.book.mark_to_market();
    }

    /// Monitor open positions' triple barriers and synthesise exit orders
    /// when TP, SL or the vertical barrier is breached.
    fn check_barriers(&mut self, now: DateTime<Utc>) {
        let monitored = self.book.barrier_positions();
        for (strategy_id, symbol, qty, is_long, barrier) in monitored {
            let Some(mark) = self.book.mark(&symbol) else {
                continue;
            };
            let exit = barrier.check(mark, now);
            if !exit.should_exit {
                continue;
            }
            self.barrier_seq += 1;
            let exit_id = OrderId::new(format!(
                "barrier-{}-{}",
                self.barrier_seq,
                UuidLike(&symbol)
            ));
            let side = if is_long {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            };
            let asset_id = strategy_id.split('@').nth(1).unwrap_or(&symbol).to_string();
            let request = OrderRequest {
                id: exit_id.clone(),
                symbol: symbol.clone(),
                side,
                order_type: OrderType::Market,
                qty: qty.abs(),
                price: None,
                time_in_force: crate::types::TimeInForce::Gtc,
                reduce_only: true,
                strategy_id: strategy_id.clone(),
                asset_id,
                barrier: None, // exits never carry a barrier (no recursion)
            };

            // Record and immediately fill at the barrier exit price.
            let mut order = Order::new(exit_id.clone(), request.clone(), now);
            order.status = OrderStatus::Accepted;
            order.updated_at = now;
            self.orders.insert(exit_id.clone(), order);
            self.events.push(ExecutionEvent::OrderAccepted {
                order_id: exit_id.clone(),
                symbol: symbol.clone(),
                at: now,
            });

            let fee = self.config.fee_bps / 10_000.0 * exit.exit_price * qty.abs();
            let applied = self.book.apply_fill(
                &strategy_id,
                &symbol,
                side,
                qty.abs(),
                exit.exit_price,
                fee,
                None,
            );
            if applied.is_err() {
                // A closing fill cannot fail capacity checks; log loudly if
                // it ever does rather than swallowing it.
                tracing::error!(
                    %strategy_id,
                    %symbol,
                    "barrier exit fill failed unexpectedly: {applied:?}"
                );
                continue;
            }
            let order = self.orders.get_mut(&exit_id).unwrap();
            order.record_fill(qty.abs(), exit.exit_price, fee, now);
            self.events.push(ExecutionEvent::OrderFilled {
                order_id: exit_id.clone(),
                filled_qty: qty.abs(),
                fill_price: exit.exit_price,
                fee,
                at: now,
            });
            self.events.push(ExecutionEvent::Fill(FillEvent {
                order_id: exit_id,
                symbol,
                side,
                qty: qty.abs(),
                price: exit.exit_price,
                fee,
                executed_at: now,
            }));
        }
    }

    /// Drain execution events since the last call.
    pub fn drain_events(&mut self) -> Vec<ExecutionEvent> {
        std::mem::take(&mut self.events)
    }

    /// Tracked state of one order.
    pub fn order_state(&self, order_id: &OrderId) -> Option<&Order> {
        self.orders.get(order_id)
    }

    /// All tracked orders.
    pub fn orders(&self) -> Vec<&Order> {
        self.orders.values().collect()
    }

    /// Capital snapshot for a strategy instance.
    pub fn capital(&self, strategy_id: &str) -> Option<Capital> {
        self.book.capital(strategy_id)
    }
}

/// Small helper so barrier order ids embed the symbol legibly.
struct UuidLike<'a>(&'a str);
impl std::fmt::Display for UuidLike<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for c in self.0.chars() {
            if c.is_ascii_alphanumeric() {
                write!(f, "{c}")?;
            }
        }
        Ok(())
    }
}

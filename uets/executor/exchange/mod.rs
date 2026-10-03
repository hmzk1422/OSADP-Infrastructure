//! `exchange` — everything that communicates with a broker.
//!
//! Buying, selling, cancelling, marking to market, and reading the account
//! (balances, margin, fees) all live behind the three connector traits
//! re-exported here from [`exchange`]:
//!
//! ```text
//! exchange/
//!   mod.rs        # this file: wiring + the buy/sell/margin diagnostic tests
//!   exchange.rs   # VenueSpec, AccountSnapshot, ConnectorError, and the
//!                 # MarketDataConnector / ExecutionConnector / AccountConnector
//!                 # contract every venue adapter implements
//! ```
//!
//! This is the *only* seam between this process and the outside world.
//! `crate::core` is the custom-made engine (gui, telemetry, features,
//! strategy, pnl); it consumes the results of a connector call — a `Fill`,
//! an `AccountSnapshot` — and never performs one. `crate::connectors`
//! supplies the transport per venue. Nothing here knows a venue's wire format.

pub mod exchange;

pub use exchange::*;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::core::order::{Fill, Order, OrderId, OrderRequest, Side};
    use crate::core::portfolio::Portfolio;

    /// A fake broker with real settlement and no network.
    ///
    /// It exists to prove the boundary itself: buying, selling, and reading
    /// margin all happen through the traits above — never by reaching into
    /// `core` from a venue, and never by reaching into a venue from `core`.
    struct FakeBroker {
        spec: VenueSpec,
        marks: BTreeMap<String, f64>,
        /// Quote-side (USDT) wallet.
        quote_free: f64,
        /// Base-side (BTC) wallet.
        base_free: f64,
        next_order: u64,
    }

    impl FakeBroker {
        fn new(starting_cash: f64, price: f64) -> Self {
            Self {
                spec: VenueSpec::new("fake", "USDT").with_fees(0.0002, 0.0004),
                marks: BTreeMap::from([("BTCUSDT".to_string(), price)]),
                quote_free: starting_cash,
                base_free: 0.0,
                next_order: 0,
            }
        }
    }

    impl Venue for FakeBroker {
        fn venue(&self) -> VenueSpec {
            self.spec.clone()
        }
    }

    impl MarketDataConnector for FakeBroker {
        fn mark_price(&self, symbol: &str) -> Result<f64, ConnectorError> {
            self.marks
                .get(symbol)
                .copied()
                .ok_or_else(|| ConnectorError::rejected(format!("no market for {symbol}")))
        }
    }

    impl ExecutionConnector for FakeBroker {
        fn submit(&mut self, request: &OrderRequest) -> Result<Vec<Fill>, ConnectorError> {
            request.validate()?;
            let price = self.mark_price(&request.symbol)?;
            let notional = price * request.qty;
            let fee = self.spec.taker_fee * notional;
            match request.side {
                Side::Buy => {
                    self.quote_free -= notional + fee;
                    self.base_free += request.qty;
                }
                Side::Sell => {
                    self.quote_free += notional - fee;
                    self.base_free -= request.qty;
                }
            }
            self.next_order += 1;
            Ok(vec![Fill::new(
                OrderId::new(format!("fake-{}", self.next_order)),
                request.symbol.clone(),
                request.side,
                request.qty,
                price,
                fee,
            )])
        }

        fn cancel(&mut self, _symbol: &str, _order_id: &OrderId) -> Result<(), ConnectorError> {
            Ok(())
        }

        fn open_orders(&mut self, _symbol: Option<&str>) -> Result<Vec<Order>, ConnectorError> {
            Ok(Vec::new())
        }
    }

    impl AccountConnector for FakeBroker {
        fn snapshot(&self) -> Result<AccountSnapshot, ConnectorError> {
            let mut snapshot = AccountSnapshot::new("fake");
            snapshot.can_trade = true;
            snapshot.maker_fee = self.spec.maker_fee;
            snapshot.taker_fee = self.spec.taker_fee;
            snapshot.balances.insert(
                "USDT".to_string(),
                AssetBalance {
                    free: self.quote_free,
                    locked: 0.0,
                },
            );
            snapshot.balances.insert(
                "BTC".to_string(),
                AssetBalance {
                    free: self.base_free,
                    locked: 0.0,
                },
            );
            Ok(snapshot)
        }
    }

    /// The concrete type satisfies the full [`ExchangeConnector`] bound.
    fn is_a_complete_exchange() {
        fn assert<T: ExchangeConnector>() {}
        assert::<FakeBroker>();
    }

    #[test]
    fn a_fake_broker_is_a_complete_exchange() {
        is_a_complete_exchange();
    }

    #[test]
    fn buying_and_selling_go_through_the_exchange_contract() {
        let mut broker = FakeBroker::new(10_000.0, 100.0);
        let mut portfolio = Portfolio::new(10_000.0, 3.0).unwrap();

        // Buy: the connector fills, `core` folds the fill into its own book.
        let buy = OrderRequest::market("BTCUSDT", Side::Buy, 0.5);
        let fills = broker.submit(&buy).expect("a buy must fill at market");
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, Side::Buy);
        portfolio.apply_fill(&fills[0]);

        let marks = broker.marks(&["BTCUSDT".to_string()]).unwrap();
        let size = portfolio
            .holdings()
            .open_holdings()
            .find(|(symbol, _)| symbol.as_str() == "BTCUSDT")
            .map(|(_, holding)| holding.size)
            .expect("the buy opened a long");
        assert_eq!(size, 0.5);

        // Sell: same path, opposite side, back to flat.
        let sell = OrderRequest::market("BTCUSDT", Side::Sell, 0.5);
        let fills = broker.submit(&sell).expect("a sell must fill at market");
        assert_eq!(fills[0].side, Side::Sell);
        portfolio.apply_fill(&fills[0]);
        assert!(
            portfolio.holdings().open_holdings().next().is_none(),
            "the round trip closed the book"
        );

        // The broker's wallet moved with the fills: notional + fee out on the
        // buy, back in (minus the sell fee) on the close, netting to one fee.
        let one_fee = 0.0004 * 100.0 * 0.5;
        let expected = 10_000.0 - 2.0 * one_fee;
        assert!((broker.quote_free - expected).abs() < 1e-9);
        assert_eq!(broker.base_free, 0.0, "flat after the round trip");

        // Core-side equity still reconciles against the marks it was given.
        assert!(portfolio.equity(&marks).is_finite());
    }

    #[test]
    fn broker_margin_is_read_from_the_account_connector() {
        let broker = FakeBroker::new(10_000.0, 100.0);
        let snapshot = broker.snapshot().expect("the fake broker has an account");
        assert!(snapshot.can_trade, "the venue says trading is allowed");
        assert_eq!(snapshot.free("USDT"), 10_000.0);
        assert_eq!(snapshot.total("USDT"), 10_000.0, "nothing is locked");
        assert_eq!(snapshot.venue, "fake");

        // `core` computes its own margin view from the same numbers without
        // ever calling the broker itself.
        let mut portfolio = Portfolio::new(10_000.0, 3.0).unwrap();
        let marks = BTreeMap::from([("BTCUSDT".to_string(), 100.0)]);
        assert_eq!(portfolio.margin_used(&marks), 0.0, "a flat book uses none");
        assert_eq!(portfolio.available_margin(&marks), 10_000.0);

        // Open 1 BTC long: margin = gross / leverage, available = equity - margin.
        let fill = Fill::new(OrderId::new("a"), "BTCUSDT", Side::Buy, 1.0, 100.0, 0.0);
        portfolio.apply_fill(&fill);
        assert!((portfolio.margin_used(&marks) - 100.0 / 3.0).abs() < 1e-9);
        assert_eq!(
            portfolio.available_margin(&marks),
            portfolio.equity(&marks) - portfolio.margin_used(&marks)
        );
    }

    #[test]
    fn an_unknown_market_is_a_rejection_not_a_panic() {
        let broker = FakeBroker::new(10_000.0, 100.0);
        let error = broker.mark_price("DOGEUSDT").unwrap_err();
        assert!(!error.is_retryable(), "no market is a permanent answer");
    }
}

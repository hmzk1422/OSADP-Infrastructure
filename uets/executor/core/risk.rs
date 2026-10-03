//! Pre-trade limits — one set of guardrails for every venue.
//!
//! The check runs in `core`, before a request reaches `connectors/<venue>`, so
//! a limit can never be enforced on Binance and forgotten on Bybit. Each breach
//! is a named variant, because "order rejected" with no reason is how a live
//! system quietly stops trading.

use std::collections::BTreeMap;
use std::fmt;

use super::order::{Fill, OrderError, OrderId, OrderRequest};
use super::portfolio::Portfolio;

/// Ceilings a single order, and the book it would create, must respect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskLimits {
    /// Absolute size allowed in one symbol, in base units.
    pub max_position_qty: f64,
    /// Absolute notional allowed in one symbol, in quote currency.
    pub max_notional: f64,
    /// Gross exposure / equity ceiling for the whole book.
    pub max_leverage: f64,
    /// Peak-to-trough equity fall at which new risk is refused.
    pub max_drawdown_frac: f64,
}

impl Default for RiskLimits {
    /// Wide but finite. `f64::INFINITY` is indistinguishable from "no limit",
    /// and the point of this type is that it always says no *somewhere*.
    fn default() -> Self {
        Self {
            max_position_qty: 1_000.0,
            max_notional: 1_000_000.0,
            max_leverage: 10.0,
            max_drawdown_frac: 0.5,
        }
    }
}

impl RiskLimits {
    /// Refuses non-positive or non-finite ceilings at construction, so a
    /// misconfigured limit fails at startup rather than on the first order.
    pub fn new(
        max_position_qty: f64,
        max_notional: f64,
        max_leverage: f64,
        max_drawdown_frac: f64,
    ) -> Result<Self, String> {
        for (name, value) in [
            ("max_position_qty", max_position_qty),
            ("max_notional", max_notional),
            ("max_leverage", max_leverage),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive, got {value}"));
            }
        }
        if !max_drawdown_frac.is_finite() || max_drawdown_frac <= 0.0 || max_drawdown_frac > 1.0 {
            return Err(format!(
                "max_drawdown_frac must be in (0, 1], got {max_drawdown_frac}"
            ));
        }
        Ok(Self {
            max_position_qty,
            max_notional,
            max_leverage,
            max_drawdown_frac,
        })
    }
}

/// Why an order was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum RiskBreach {
    /// The request itself was malformed; carries the underlying reason.
    Malformed(OrderError),
    /// No mark for this symbol, so the notional cannot be judged.
    UnknownMark {
        symbol: String,
    },
    PositionLimit {
        resulting: f64,
        limit: f64,
    },
    NotionalLimit {
        resulting: f64,
        limit: f64,
    },
    LeverageLimit {
        resulting: f64,
        limit: f64,
    },
    DrawdownLimit {
        current: f64,
        limit: f64,
    },
}

impl fmt::Display for RiskBreach {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(e) => write!(f, "malformed order: {e}"),
            Self::UnknownMark { symbol } => {
                write!(f, "no mark price for {symbol}, refusing to size the order")
            }
            Self::PositionLimit { resulting, limit } => {
                write!(f, "order would leave {resulting} units, limit {limit}")
            }
            Self::NotionalLimit { resulting, limit } => {
                write!(f, "notional {resulting:.2} over limit {limit:.2}")
            }
            Self::LeverageLimit { resulting, limit } => {
                write!(f, "leverage {resulting:.2}x over limit {limit:.2}x")
            }
            Self::DrawdownLimit { current, limit } => {
                write!(f, "drawdown {current:.4} past limit {limit:.4}")
            }
        }
    }
}

impl std::error::Error for RiskBreach {}

/// Check `request` against `limits` and the book it would create.
///
/// Check order matters: a malformed request and a blown account are both
/// refused before anything is sized, because neither has a meaningful resulting
/// size to compare against a ceiling.
pub fn check(
    limits: &RiskLimits,
    portfolio: &Portfolio,
    marks: &BTreeMap<String, f64>,
    request: &OrderRequest,
) -> Result<(), RiskBreach> {
    request.validate().map_err(RiskBreach::Malformed)?;

    let drawdown = portfolio.ledger().max_drawdown();
    if drawdown > limits.max_drawdown_frac {
        return Err(RiskBreach::DrawdownLimit {
            current: drawdown,
            limit: limits.max_drawdown_frac,
        });
    }

    let mark = *marks
        .get(&request.symbol)
        .ok_or_else(|| RiskBreach::UnknownMark {
            symbol: request.symbol.clone(),
        })?;

    let holding = portfolio.holdings().holding(&request.symbol);
    let resulting_size = holding.size + request.side.signed(request.qty);
    if resulting_size.abs() > limits.max_position_qty {
        return Err(RiskBreach::PositionLimit {
            resulting: resulting_size.abs(),
            limit: limits.max_position_qty,
        });
    }
    let resulting_notional = resulting_size.abs() * mark;
    if resulting_notional > limits.max_notional {
        return Err(RiskBreach::NotionalLimit {
            resulting: resulting_notional,
            limit: limits.max_notional,
        });
    }

    // Leverage is a property of the whole book *after* the fill, not of this
    // order alone, so the check runs on a throwaway copy filled at the mark.
    let mut probe = portfolio.clone();
    probe.apply_fill(&Fill::new(
        OrderId::new("risk-probe"),
        request.symbol.clone(),
        request.side,
        request.qty,
        mark,
        0.0,
    ));
    let leverage = probe.effective_leverage(marks);
    if leverage > limits.max_leverage {
        return Err(RiskBreach::LeverageLimit {
            resulting: leverage,
            limit: limits.max_leverage,
        });
    }
    Ok(())
}

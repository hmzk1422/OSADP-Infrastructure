//! Binance spot: the balance view used for reconciliation.
//!
//! The job of this file is one translation: `/api/v3/account` returns *every*
//! asset the venue has ever listed — several hundred rows, nearly all zero —
//! while reconciliation only needs the assets that carry value. Rows whose free
//! and locked quantities are both zero are therefore dropped, and the contract
//! published to `core` is: **an asset absent from `balances` has balance zero**.
//!
//! Fees arrive in basis points (`makerCommission: 10` is 0.10%) and are divided
//! by 10 000 here. That is the only arithmetic in the file, and the test below
//! pins it, because getting it wrong would scale every fee in the ledger by
//! four orders of magnitude.
//!
//! What this adapter deliberately does not model: `can_withdraw`/`can_deposit`.
//! Neither affects an order, and inventing a meaning for them would be worse
//! than leaving them out. `can_trade` *does* affect an order, so it is kept.

use binance::account::Account;
use binance::api::Binance;
use binance::model::{AccountInformation, Balance};

use crate::exchange::{
    AccountConnector, AccountSnapshot, AssetBalance, ConnectorError, Venue, VenueSpec,
};

use super::trade::map_error;

/// Binance spot account view over REST.
pub struct BinanceAccount {
    account: Account,
    spec: VenueSpec,
}

impl BinanceAccount {
    pub fn new(api_key: Option<String>, secret_key: Option<String>) -> Self {
        Self {
            account: Binance::new(api_key, secret_key),
            spec: super::spot_spec(),
        }
    }

    /// Point the client at testnet, whose balances are not real money.
    pub fn testnet(mut self, on: bool) -> Self {
        self.account.set_testnet(on);
        self
    }

    /// Replace the declared schedule — e.g. with fees read from a snapshot.
    pub fn with_spec(mut self, spec: VenueSpec) -> Self {
        self.spec = spec;
        self
    }

    /// The venue's raw answer, for anything this adapter does not model.
    pub fn raw_account(&self) -> Result<AccountInformation, ConnectorError> {
        self.account.get_account().map_err(map_error)
    }
}

impl Venue for BinanceAccount {
    fn venue(&self) -> VenueSpec {
        self.spec.clone()
    }
}

impl AccountConnector for BinanceAccount {
    fn snapshot(&self) -> Result<AccountSnapshot, ConnectorError> {
        let information = self.raw_account()?;
        Ok(snapshot_from_account(
            &information,
            self.spec.name,
            &self.spec,
        ))
    }
}

/// The venue's answer as an [`AccountSnapshot`].
///
/// Pure, so the whole translation — including the basis-point conversion and
/// the zero-row filter — is testable without a network.
pub fn snapshot_from_account(
    information: &AccountInformation,
    venue: &'static str,
    fallback: &VenueSpec,
) -> AccountSnapshot {
    let balances = information
        .balances
        .iter()
        .filter(|balance| !is_empty(balance))
        .map(|balance| (balance.asset.clone(), asset_balance(balance)))
        .collect();
    // A venue reporting 0/0 has said nothing useful, and the declared schedule
    // is the better guess. When they disagree, overstating the fee understates
    // pnl — the direction worth being wrong in.
    let reports_fees = information.maker_commission > 0.0 || information.taker_commission > 0.0;
    let (maker_fee, taker_fee) = if reports_fees {
        (
            fee_from_basis_points(information.maker_commission),
            fee_from_basis_points(information.taker_commission),
        )
    } else {
        (fallback.maker_fee, fallback.taker_fee)
    };
    AccountSnapshot {
        balances,
        can_trade: information.can_trade,
        maker_fee,
        taker_fee,
        ..AccountSnapshot::new(venue)
    }
}

/// Binance quotes fees in basis points: 10 means 0.10%.
pub fn fee_from_basis_points(basis_points: f32) -> f64 {
    basis_points as f64 / 10_000.0
}

/// Parse one of the venue's fixed-width decimal strings. An unreadable value
/// is zero rather than a reason to fail the whole reconciliation.
fn parse_quantity(text: &str) -> f64 {
    text.trim().parse().unwrap_or(0.0)
}

/// One asset row, both quantities parsed leniently.
fn asset_balance(balance: &Balance) -> AssetBalance {
    AssetBalance {
        free: parse_quantity(&balance.free),
        locked: parse_quantity(&balance.locked),
    }
}

/// True when the row carries nothing. Zero exactly, not "small": an asset with
/// 1e-9 left is a balance the venue still reports and the ledger should see.
fn is_empty(balance: &Balance) -> bool {
    parse_quantity(&balance.free) == 0.0 && parse_quantity(&balance.locked) == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn information(maker: f32, taker: f32, balances: Vec<Balance>) -> AccountInformation {
        AccountInformation {
            maker_commission: maker,
            taker_commission: taker,
            buyer_commission: 0.0,
            seller_commission: 0.0,
            can_trade: true,
            can_withdraw: true,
            can_deposit: true,
            balances,
        }
    }

    fn balance(asset: &str, free: &str, locked: &str) -> Balance {
        Balance {
            asset: asset.to_string(),
            free: free.to_string(),
            locked: locked.to_string(),
        }
    }

    #[test]
    fn only_assets_carrying_value_survive_the_translation() {
        let venue = information(
            10.0,
            10.0,
            vec![
                balance("BTC", "1.50000000", "0.00000000"),
                balance("DOGE", "0.00000000", "0.00000000"),
                balance("USDT", "900.00000000", "100.00000000"),
                // Dust is still a balance: the venue reports it, so the ledger
                // must see it rather than silently agreeing with a zero.
                balance("SHIB", "0.00000001", "0.00000000"),
                // Unreadable text is treated as zero, so the row goes too.
                balance("BROKEN", "not-a-number", "0.00000000"),
            ],
        );
        let snapshot = snapshot_from_account(&venue, "binance", &super::super::spot_spec());

        assert_eq!(snapshot.venue, "binance");
        assert!(
            snapshot.can_trade,
            "can_trade decides whether an order is legal"
        );
        assert_eq!(snapshot.balances.len(), 3, "DOGE and BROKEN dropped");
        assert_eq!(snapshot.balances["BTC"].free, 1.5);
        assert_eq!(snapshot.balances["USDT"].locked, 100.0);
        assert_eq!(snapshot.balances["SHIB"].free, 1e-8);
        assert!(!snapshot.balances.contains_key("DOGE"), "absent means zero");
    }

    #[test]
    fn basis_points_become_fee_fractions() {
        // Binance's makerCommission is 10 for a 0.10% spot account.
        assert!((fee_from_basis_points(10.0) - 0.001).abs() < 1e-12);
        assert!((fee_from_basis_points(7.5) - 0.00075).abs() < 1e-12);
        assert_eq!(fee_from_basis_points(0.0), 0.0);
    }

    #[test]
    fn fees_the_venue_does_not_report_fall_back_to_the_declared_schedule() {
        let venue = information(0.0, 0.0, vec![balance("USDT", "10.0", "0.0")]);
        let spec = super::super::spot_spec();
        let snapshot = snapshot_from_account(&venue, "binance", &spec);
        assert_eq!(snapshot.maker_fee, spec.maker_fee);
        assert_eq!(snapshot.taker_fee, spec.taker_fee);

        let venue = information(10.0, 15.0, vec![balance("USDT", "10.0", "0.0")]);
        let snapshot = snapshot_from_account(&venue, "binance", &spec);
        assert!(
            (snapshot.maker_fee - 0.001).abs() < 1e-12,
            "the account's own fee"
        );
        assert!(
            (snapshot.taker_fee - 0.0015).abs() < 1e-12,
            "and its taker tier"
        );
    }
}

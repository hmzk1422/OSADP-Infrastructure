import io, sys
p = r"uets_rs/engine/src/lib.rs"
t = io.open(p, encoding="utf-8").read()
Q = chr(34)
s = []
s += ["    fn spawn_shadow(&mut self, feat: &Feature, fill_price: f64, bp: f64, side: Side) {"]
s += ["        let slip = if self.cfg.enable_slippage { self.cfg.slippage_bps_fallback } else { 0.0 };"]
s += ["        let entry_price = match side {"]
G = ["Side::Long", "Side::Short"]
s += ["            " + G[0] + " => fill_price * (1.0 + slip / 10_000.0),"]
s += ["            " + G[1] + " => fill_price * (1.0 - slip / 10_000.0),"]
s += ["        };"]
s += ["        let qty = if self.cfg.risk_pct_per_trade > 0.0 && bp > EPS && self.equity >  ulate  0.0 {"]
s += ["            let risk_usd = self.equity * self.cfg.risk_pct_per_trade;"]
s += ["            let notional = (risk_usd / bp).min(self.equity * self.cfg.max_leverage);"]
s += ["            notional / entry_price"]
s += ["        } else {"]
s += ["            self.cfg.max_position_usd / entry_price"]
s += ["        };"]
s += ["        if qty <= 0.0 { return; }"]
s += ["        let notional = qty * entry_price;"]
s += ["        let (stop,target) = match side {"]
s += ["            Side::Long => (entry_price * (1.0 - bp), entry_price * (1.0 + bp * self.cfg.rr_ratio)),"]
s += ["            Side::Short => (entry_price * (1.0 + bp), entry_price * (1.0 - bp * self.cfg.rr_ratio)),"]
s += ["        };"]
s += ["        let m = self.macro_features.unwrap_or_default();"]
s += ["        let hour = {"]
s += ["            let sod = (feat.ts / 1000).rem_euclid(86_400);"]
s += ["            (sod / 3600) as f64 + ((sod % 3600) / 60) as f64 / 60.0"]
s += ["        };"]
s += ["        let feature_snapshot = vec!["]
s += ["            m.spread_bps_z, m.ma_slope_5, m.rsi_x_ma, m.ma_accel,"]
s += ["            m.price_roc_3, m.rsi_2, m.sma_14, m.ofi_z,"]
s += ["            m.cvd_delta_z, m.cvd_slope_z, m.absorption_z, m.tick_autocorr,"]
s += ["            m.inst_ratio, m.delta_divergence_z, m.depth_imbalance_z, hour,"]
s += ["        ];"]
s += ["        self.n_voided +=1;"]
s += ["        self.shadows.push(ShadowPosition { side, entry_ms: feat.ts, entry_price,"]
s += ["            qty, notional, barrier_pct: bp, stop, target,"]
s += ["            meta_pwin: feat.meta_pwin, feature_snapshot });"]
s += ["    }"]
new_block = "\n".join(s) + "\n"
anchor = "    fn check_exit(&mut self, ts: i64, close: f64) -> Option<TradeRecord> {"
n = t.count(anchor)
if n != 1: print(f"ABORT: count={n}"); sys.exit(1)
t = t.replace(anchor, new_block + anchor)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("SPAWN OK")

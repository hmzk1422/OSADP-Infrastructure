import re, io
p = 'uets_rs/engine/src/lib.rs'
t = open(p, encoding='utf-8').read()
def sub(old, new):
    global t
    n = t.count(old)
    if n == 0: raise SystemExit(f'MISSING: {old[:60]} count=0')
    if n != 1: raise SystemExit(f'AMBIG: {old[:60]} count={n}')
    t = t.replace(old, new)

sub('regime_gate: Option<RegimeGate>,\n}',
    'regime_gate: Option<RegimeGate>,\n    closed_stats: VecDeque<RegimeTradeStat>,\n}', )
sub('regime_gate: None,\n        }',
    'regime_gate: None,\n            closed_stats: VecDeque::new(),\n        }' )
sub('let rec = TradeRecord {',
    '''// Regime-stat record (research regime_data.py parity: rolling strategy-state \\
        // / vol-prior vars use STRICTLY-PRIOR trades). Push THIS trade' s stat \\
        // BEFORE it becomes "prior" for the next entry.
        let range_pct = if p.entry_price >​ 0.0 { (p.hi - p.lo) / p.entry_price } else { f64::NAN };
        let mae_pct = if p.entry_price >​ 0.0 { (p.entry_price - p.lo) / p.entry_price } else { f64::NAN };
        let mfe_pct = if p.entry_price >​ 0.0 { (p.hi - p.entry_price) / p.entry_price } else { f64::NAN };
        let gap_min = match self.closed_stats.back() {
            Some(prev) => (p.entry_ms - prev.entry_ms) as f64 /​ 60000.0,
            None => f64::NAN,
        };
        let is_quick = if gap_min.is_nan() { 0.0 } else if gap_min <​ 6.0 { 1.0 } else { 0.0 };
        self.closed_stats.push_back(RegimeTradeStat {
            r: r_raw,
            range_pct, mae_pct, mfe_pct,
            gap_min,
            is_winner: if r_raw >​ 0.0 { 1.0 } else { 0.0 },
            is_quick,
        });
        if self.closed_stats.len() >​ 64 { self.closed_stats.pop_front(); }
        let rec = TradeRecord {''' )
t, n_repl = re.subn(re.escape('pub struct RegimeGate {') + '.*?' + re.escape('/// A hypothetical (voided) trade'),
'''pub struct GateNode {
    pub feature: i32,
    pub threshold: f64,
    pub left: i32,
    pub right: i32,
}

#[derive(Debug, Clone)]
pub struct RegimeGate {
    pub features: Vec<String>,
    pub nodes: Vec<GateNode>,
    pub favorable: Vec<bool>,
}

impl RegimeGate {
    pub fn classify(&self, x: &[f64]) -> Option<usize> {
        let mut node: i32 = 0;
        loop {
            let n = self.nodes.get(node as usize)?;
            if n.left <​ 0 || n.right <​ 0 { return Some(node as usize); }
            let v = x.get(n.feature as usize).copied()?;
            node = if v.is_nan() || v <= n.threshold { n.left } else { n.right };
        }
    }
    pub fn is_tradeable(&self, leaf: usize) -> bool {
        self.favorable.get(leaf).copied().unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
pub struct RegimeTradeStat {
    pub r: f64,
    pub range_pct: f64,
    pub mae_pct: f64,
    pub mfe_pct: f64,
    pub gap_min: f64,
    pub is_winner: f64,
    pub is_quick: f64,
}

fn _stat_mean(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if vals.is_empty() { return f64::NAN; }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn _stat_median(v: &[f64]) -> f64 {
    let mut vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if vals.is_empty() { return f64::NAN; }
    vals.sort_by(|a,b| a.total_cmp(b));
    let n = vals.len();
    if n % 2 == 1 { vals[n/ϛ2] } else { (vals[n/ϛ2-1] + vals[n/ϛ2]) / 2.0 }
}

fn _stat_std(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if vals.len() <​ 2 { return f64::NAN; }
    let n = vals.len() as f64;
    let mu = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x-mu)*(x-mu)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// A hypothetical (voided) trade''', t, flags=re.S|re.M)
if n_repl != 1: raise SystemExit(f'RE1 count={n_repl}')

# try_enter: replace gate block with tree + else
s1 = '// Regime-detection gate'
s2 = 'if self.labeller.is_some()'
new_enter = '''// Regime-detection gate (the research tree; checked FIRST - it sits between
        // the main executor and execution, exactly as specified). The gate is the
        // ONLY metalabeller when deployed: the LGBM labeller below is skipped so
        // there are exactly 2 layers - main executor -> regime gate -> execution.
        // A non-tradeable leaf voids the candidate (SHADOW-recorded, never lost).
        if let Some(gate) = self.regime_gate.as_ref() {
            let x = self.regime_feature_vector(&feat, fill_price);
            if let Some(leaf) = gate.classify(&x) {
                if !gate.is_tradeable(leaf) {
                    self.last_entry_block = "regime_filter";
                    self.spawn_shadow(&feat, fill_price, bp, side);
                    return;
                }
            }
        } else '''
t, n2 = re.subn(re.escape(s1) + '.*?' + re.escape(s2), new_enter + s2, t, flags=re.S|re.M)
if n2 != 1: raise SystemExit(f'RE2 count={n2}')

# macro_feature_vector -> regime_feature_vector + rolling helpers
s3 = '/// The 15-feature macro vector'
s4 = 'pub fn model_loaded'
new_vec = '''/// Build the regime gate feature vector in the SAME column order as the
    /// exported research tree (regime_trees.py model_features(): ENTRY_STATE +
    /// price_rel_sma14 + vol_prior20{range,mae} + hour_of_day + 15 rolling
    /// strategy-state vars). Every base value is computed live from the same
    /// 1-second/macro state the main model uses - no future data, no
    /// rewrites. NaN features route to the tree' s left child (exactly how
    /// sklearn handles missing values), matching research warm-up rows.
    fn _rr(&self, key: &str, window: usize, min_req: usize) -> f64 {
        let mut vals: Vec<f64> = Vec::new();
        for s in self.closed_stats.iter().rev().take(window) {
            vals.push(match key {
                "r" => s.r, "range_pct" => s.range_pct, "mae_pct" => s.mae_pct,
                "mfe_pct" => s.mfe_pct, "gap_min" => s.gap_min, "is_winner" => s.is_winner,
                "is_quick" => s.is_quick,
                _ => return f64::NAN,
            });
        }
        vals.retain(|x| !x.is_nan());
        if vals.len() <​ min_req { return f64::NAN; }
        match key {
            "r" | "range_pct" | "mae_pct" | "mfe_pct" | "gap_min" | "is_winner" | "is_quick" => _stat_mean(&vals),
            _ => _stat_mean(&vals),
        }
    }
    fn _rm(&self, key: &str, window: usize, min_req: usize) -> f64 {
        let mut vals: Vec<f64> = Vec::new();
        for s in self.closed_stats.iter().rev().take(window) {
            vals.push(match key {
                "r" => s.r, "range_pct" => s.range_pct, "mae_pct" => s.mae_pct,
                "mfe_pct" => s.mfe_pct, "gap_min" => s.gap_min,
                _ => return f64::NAN,
            });
        }
        vals.retain(|x| !x.is_nan());
        if vals.len() <​ min_req { return f64::NAN; }
        _stat_median(&vals)
    }
    fn _rs(&self, key: &str, window: usize, min_req: usize) -> f64 {
        let mut vals: Vec<f64> = Vec::new();
        for s in self.closed_stats.iter().rev().take(window) {
            vals.push(if key == "r" { s.r } else { return f64::NAN; } });
        }
        vals.retain(|x| !x.is_nan());
        if vals.len() <​ min_req { return f64::NAN; }
        _stat_std(&vals)
    }
    fn _pf(&self, window: usize, min_req: usize) -> f64 {
        let mut vals: Vec<f64> = Vec::new();
        for s in self.closed_stats.iter().rev().take(window) { vals.push(s.r); }
        vals.retain(|x| !x.is_nan());
        if vals.len() <​ min_req { return f64::NAN; }
        let w: f64 = vals.iter().filter(|&&x| x >​ 0.0).sum();
        let l: f64 = vals.iter().filter(|&&x| x <​ 0.0).map(|&x| -x).sum();
        if l >​ 1e-9 { w / l } else { f64::NAN }
    }
    pub fn regime_feature_vector(&self, feat: &Feature, fill_price: f64) -> Vec<f64> {
        let m = self.macro_features.unwrap_or_default();
        let nan = f64::NAN;
        let mut x: Vec<f64> = Vec::with_capacity(28);
        let names: Vec<String> = self.regime_gate.as_ref().map(|g| g.features.clone()).unwrap_or_default();
        for name in names {
            match name.as_str() {
                "spread_bps_z_open" => x.push(m.spread_bps_z),
                "ma_slope_5_open" => x.push(m.ma_slope_5),
                "ma_accel_open" => x.push(m.ma_accel),
                "price_roc_3_open" => x.push(m.price_roc_3),
                "rsi_2_open" => x.push(m.rsi_2),
                "cvd_slope_z_open" => x.push(m.cvd_slope_z),
                "tick_autocorr_open" => x.push(m.tick_autocorr),
                "inst_ratio_open" => x.push(m.inst_ratio),
                "depth_imbalance_z_open" => x.push(m.depth_imbalance_z),
                "price_rel_sma14" => x.push(if m.sma_14 >​ 0.0 { fill_price / m.sma_14 - 1.0 } else { nan }),
                "vol_prior20_range" => x.push(self._rm("range_pct", 20, 20)),
                "vol_prior20_mae" => x.push(self._rm("mae_pct",  ulate 20,  ulate 20)),
                "hour_of_day" => {
                    let sod = (feat.ts /​ 1000).rem_euclid(86_400);
                    x.push((sod /​ 3600) as f64 + ((sod %​ 3600) /​ 60) as f64 /​ 60.0);
                }
                "strat_prev10_mean_r" => x.push(self._rr("r", 10, 10)),
                "strat_prev10_median_r" => x.push(self._rm("r", 10, 10)),
                "strat_prev10_win_rate" => x.push(self._rr("is_winner", 10, 10)),
                "strat_prev20_win_rate" => x.push(self._rr("is_winner", 20, 10)),
                "strat_prev50_win_rate" => x.push(self._rr("is_winner", 50,, 10)),
                "strat_prev20_profit_factor" => x.push(self._pf(20,  ulate 10)),
                "strat_prev20_ret_std" => x.push(self._rs("r", 20,, 10)),
                "strat_prev10_mae" => x.push(self._rr("mae_pct",  10,, 10)),
                "strat_prev10_mfe" => x.push(self._rr("mfe_pct",  10,, 10)),
                "strat_prev10_median_gap" => x.push(self._rm("gap_min", 10,, 10)),
                "strat_prev20_median_gap" => x.push(self._rm("gap_min", 20,,, 10)),
                "strat_prev50_median_gap" => x.push(self._rm("gap_min", 50,,, 10)),
                "strat_prev10_quick_frac" => x.push(self._rr("is_quick", 10,,, 10)),
                "strat_prev20_quick_frac" => x.push(self._rr("is_quick", 20,,, 10)),
                "strat_prev10_quick_winless_frac" => {
                    let vals: Vec<f64> = self.closed_stats.iter().rev().take(10.map(|s| s.is_quick * (1.0 - s.is_winner)).collect();
                    x.push(if vals.len() <​ 10 { nan } else { _stat_mean(&vals) } });
                }
                _ => x.push(nan),
            }
        }
        x
    }

    ///
    pub fn model_loaded'''
t, n3 = re.subn(re.escape(s3) + '.*?' + re.escape(s4), new_vec + s4, t, flags=re.S|re.M)
if n3 != 1: raise SystemExit(f'RE3 count={n3}')
open(p,'w', encoding='utf-8', newline='').write(t)
print('ENGINE PATCH OK')

# -*- coding: utf-8 -*-
import io, sys
p = "uets_rs/engine/src/lib.rs"
t = io.open(p, encoding="utf-8").read()
steps = []
def rep(old, new, tag):
    global t
    n = t.count(old)
    if n != 1:
        print(f"ABORT[{tag}]: count={n}")
        sys.exit(1)
    t = t.replace(old, new)
    print(f"ok:[{tag}]")

# A structs block before TradeRecord doc
rep("/// Closed-trade record; `reason`",
"""/// A virtual copy of a candidate voided by the regime gate (data collection
/// only - never touches equity/stats/lockout).
#[derive(Debug, Clone)]
pub struct ShadowPosition {
    pub side: Side,
    pub entry_ms: i64,
    pub entry_price: f64,
    pub qty: f64,
    pub notional: f64,
    pub barrier_pct: f64,
    pub stop: f64,
    pub target: f64,
    pub meta_pwin: f64,
    pub feature_snapshot: Vec<f64>,
}

/// A voided candidate closed on the 1s close series - persisted for retraining.
#[derive(Debug, Clone)]
pub struct VoidedTrade {
    pub entry_ms: i64,
    pub exit_ms: i64,
    pub side: Side,
    pub entry_price: f64,
    pub exit_price: f64,
    pub reason: String,
    pub qty: f64,
    pub notional: f64,
    pub barrier_pct: f64,
    pub r_multiple: f64,
    pub r_raw: f64,
    pub duration_s: f64,
    pub meta_pwin: f64,
    pub feature_snapshot: Vec<f64>,
}

/// Regime gate = research DecisionTree export (build/regime_gate.json).
#[derive(Debug, Clone)]
pub struct GateNode {
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
            if n.left < 0 || n.right < 0 { return Some(node as usize); }
            let v = x.get(n.feature as usize).copied()?;
            node = if v.is_nan() || v <= n.threshold { n.left } else { n.right };
        }
    }
    pub fn is_tradeable(&self, leaf: usize) -> bool {
        self.favorable.get(leaf).copied().unwrap_or(false)
    }
}

/// One closed trade's stats (strictly-prior ring for the gate's rolling features).
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

/// Closed-trade record; `reason`"", "A1")

# B enum variant
rep("Exited(TradeRecord),\n}\n", "Exited(TradeRecord),\n    /// A gate-rejected candidate closed its shadow window - data collection only.\n    VoidClosed(VoidedTrade),\n}\n", "B1")

# C snapshot field
rep("pub last_features: Option<Feature>,\n}\n", "pub last_features: Option<Feature>,\n    /// Count of voided (gate-rejected) candidates recorded.\n    pub n_voided: u64,\n}\n", "C1")

# D struct fields
rep("meta_threshold: f64,\n", "meta_threshold: f64,\n    /// Regime-detection gate (research tree; THE metalabeller when deployed).\n    regime_gate: Option<RegimeGate>,\n    /// Strictly-prior closed-trade stats for the gate's rolling features.\n    closed_stats: VecDeque<RegimeTradeStat>,\n    /// Voided (gate-rejected) candidate recording stats.\n    n_voided: u64,\n    shadows: Vec<ShadowPosition>,\n    voided: Vec<VoidedTrade>,\n", "D1")

# E new inits
rep("labeller: None,\n            meta_threshold: 0.0,\n", "labeller: None,\n            meta_threshold: 0.0,\n            regime_gate: None,\n            closed_stats: VecDeque::new(),\n            n_voided: 0,\n            shadows: Vec::new(),\n            voided: Vec::new(),\n", "E1")

# F snapshot init
rep("last_features: self.last_features,\n        }\n    }\n", "last_features: self.last_features,\n            n_voided: self.n_voided,\n        }\n    }\n", "F1")

# G set_regime_gate after set_labeller
rep("pub fn model_loaded(&self) -> bool {\n", "pub fn set_regime_gate(&mut self, gate: Option<RegimeGate>) {\n        self.regime_gate = gate;\n    }\n\n    /// True if a real LightGBM model is installed and will be scored on the\n    pub fn model_loaded(&self) -> bool {\n", "G1")

t = io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ALL OK - written")

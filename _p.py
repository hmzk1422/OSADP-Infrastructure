import io, sys
p = r"uets_rs/engine/src/lib.rs"
t = io.open(p, encoding="utf-8").read()
def rep(old, new, tag):
    global t
    n = t.count(old)
    if n != 1: print(f"ABORT[{tag}]: count={n}"); sys.exit(1)
    t = t.replace(old, new)
    print(f"ok:[{tag}]")

B = []
B += ["pub struct ShadowPosition {", "    pub side: Side,", "    pub entry_ms: i64,", "    pub entry_price: f64,", "    pub qty: f64,", "    pub notional: f64,", "    pub barrier_pct: f64,", "    pub stop: f64,", "    pub target: f64,", "    pub meta_pwin: f64,", "    pub feature_snapshot: Vec<f64>,", "}"]
B += ["pub struct VoidedTrade {", "    pub entry_ms: i64,", "    pub exit_ms: i64,", "    pub side: Side,", "    pub entry_price: f64,", "    pub exit_price: f64,", "    pub reason: String,", "    pub qty: f64,", "    pub notional: f64,", "    pub barrier_pct: f64,", "    pub r_multiple: f64,", "    pub r_raw: f64,", "    pub duration_s: f64,", "    pub meta_pwin: f64,", "    pub feature_snapshot: Vec<f64>,", "}"]
B += ["pub struct GateNode {", "    pub feature: i32,", "    pub threshold: f64,", "    pub left: i32,", "    pub right: i32,", "}"]
B += ["#[derive(Debug, Clone)]", "pub struct RegimeGate {", "    pub features: Vec<String>,", "    pub nodes: Vec<GateNode>,", "    pub favorable: Vec<bool>,", "}"]
B += ["impl RegimeGate {", "    pub fn classify(&self, x: &[f64]) -> Option<usize> {", "        let mut node: i32 = 0;", "        loop {", "            let n = self.nodes.get(node as usize)?;", "            if n.left < 0 || n.right < 0 { return Some(node as usize); }", "            let v = x.get(n.feature as usize).copied()?;", "            node = if v.is_nan() || v <= n.threshold { n.left } else { n.right };", "        }", "    }", "    pub fn is_tradeable(&self, leaf: usize) -> bool {", "        self.favorable.get(leaf).copied().unwrap_or(false)", "    }", "}"]
B += ["#[derive(Debug, Clone)]", "pub struct RegimeTradeStat {", "    pub r: f64,", "    pub range_pct: f64,", "    pub mae_pct: f64,", "    pub mfe_pct: f64,", "    pub gap_min: f64,", "    pub is_winner: f64,", "    pub is_quick: f64,", "}"]
new_block = "\n".join(B) + "\n"
rep("/// Closed-trade record; `reason`", new_block + "/// Closed-trade record; `reason`", "A")
rep("Exited(TradeRecord),\n}\n", "Exited(TradeRecord),\n    VoidClosed(VoidedTrade),\n}\n", "B")
rep("pub last_features: Option<Feature>,\n}\n", "pub last_features: Option<Feature>,\n    pub n_voided: u64,\n}\n", "C")
rep("meta_threshold: f64,\n", "meta_threshold: f64,\n    regime_gate: Option<RegimeGate>,\n    closed_stats: VecDeque<RegimeTradeStat>,\n    n_voided: u64,\n    shadows: Vec<ShadowPosition>,\n    voided: Vec<VoidedTrade>,\n", "D")
rep("labeller: None,\n            meta_threshold: 0.0,\n", "labeller: None,\n            meta_threshold: 0.0,\n            regime_gate: None,\n            closed_stats: VecDeque::new(),\n            n_voided:  ulate 0,\n            shadows: Vec::new(),\n            voided: Vec::new(),\n", "E")
rep("last_features: self.last_features,\n        }\n    }\n", "last_features: self.last_features,\n            n_voided: self.n_voided,\n        }\n    }\n", "F")
rep("pub fn model_loaded(&self) -> bool {\n", "pub fn set_regime_gate(&mut self, gate: Option<RegimeGate>) {\n        self.regime_gate = gate;\n    }\n\n    /// True if a real LightGBM model is installed and will be scored on the\n    pub fn model_loaded(&self) -> bool {\n", "G")
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("PASS1 OK")

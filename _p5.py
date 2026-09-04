import io, sys
p = r"uets_rs/engine/src/lib.rs"
t = io.open(p, encoding="utf-8").read()
old = """        if self.labeller.is_some()
            && self.meta_threshold > 0.0
            && feat.meta_pwin + 1e-12 < self.meta_threshold
        {
            self.last_entry_block = "meta_filter";
            return;
        }
"""
new = """        // Regime-detection gate (research tree; checked FIRST - THE metalabeller).
        if let Some(gate) = self.regime_gate.as_ref() {
            let x = self.regime_feature_vector(&feat, fill_price);
            if let Some(leaf) = gate.classify(&x) {
                if !gate.is_tradeable(leaf) {
                    self.last_entry_block = "regime_filter";
                    self.spawn_shadow(&feat, fill_price, bp, side);
                    return;
                }
            }
        } else if self.labeller.is_some()
            && self.meta_threshold > 0.0
            && feat.meta_pwin + 1e-12 < self.meta_threshold
        {
            self.last_entry_block = "meta_filter";
            self.spawn_shadow(&feat, fill_price, bp, side);
            return;
        }
"""
n = t.count(old)
if n != 1: print(f"ABORT: count={n}"); sys.exit(1)
t = t.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ENTER OK")

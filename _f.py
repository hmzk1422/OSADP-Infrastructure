import io, sys
p = r"uets_rs/engine/src/lib.rs"
lines = io.open(p, encoding="utf-8").read().split("\n")
for k,l in enumerate(lines):
    if ("let qty = if self.cfg" in l) and ("\nulate" in l or "ulate" in l):
        lines[k] = "        let qty = if self.cfg.risk_pct_per_trade > 0.0 && bp > EPS && self.equity > 0.0 {"
        print("fixed 1345"); break
else:
    print("ABORT: 1345 not found"); sys.exit(1)
io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
print("OK")

import io, sys
p = r"uets_rs/engine/src/lib.rs"
lines = io.open(p, encoding="utf-8").read().split("\n")
i = [k for k,l in enumerate(lines) if "pub struct GateNode" in l and "{" not in l]
if len(i) != 1: print(f"ABORT: hits={len(i)}"); sys.exit(1)
lines[i[0]] = "pub struct GateNode {"
io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
print("G-BRACE OK")

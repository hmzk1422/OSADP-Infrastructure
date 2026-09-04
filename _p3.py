import io, sys
p = r"uets_rs/engine/src/lib.rs"
lines = io.open(p, encoding="utf-8").read().split("\n")
for old, new, tag in [("pub struct VoidedTrade", "pub struct VoidedTrade {", "V"), (">pub struct ShadowPosition", "pub struct ShadowPosition {", "S")]:
    old = old.lstrip(">")
    i = [k for k,l in enumerate(lines) if old in l and "{" not in l]
    if len(i) != 1: print(f"ABORT[{tag}]: hits={len(i)}"); sys.exit(1)
    lines[i[0]] = new
    print(f"ok:[{tag}]")
io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
print("BRACE2 OK")

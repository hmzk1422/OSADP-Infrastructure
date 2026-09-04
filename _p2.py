import io, sys
p = r"uets_rs/engine/src/lib.rs"
lines = io.open(p, encoding="utf-8").read().split("\n")
def repi(old, new, tag):
    global lines
    i = [k for k,l in enumerate(lines) if old in l]
    if len(i) != 1: print(f"ABORT[{tag}]: hits={len(i)}"); sys.exit(1)
    lines[i[0]] = new
    print(f"ok:[{tag}]")
repi("pub struct GateNode", "#[derive(Debug, Clone)]\npub struct GateNode", "DG")
repi("pub struct VoidedTrade", "#[derive(Debug, Clone)]\npub struct VoidedTrade", "DV")
repi("pub struct ShadowPosition", "#[derive(Debug, Clone)]\npub struct ShadowPosition", "DS")
j = None
for k,l in enumerate(lines):
    if l.strip().startswith("n_voided:") and "ulate" in l:
        j = k; break
if j is None: print("ABORT: no ulate line"); sys.exit(1)
lines[j] = "            n_voided:  ulate  0,".replace("  ulate  ", " ")
print("ok:[U]")
io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
print("PASS2 OK")

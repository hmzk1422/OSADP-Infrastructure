import io, re, sys
p = "uets_rs/engine/src/lib.rs"
ls = io.open(p, encoding="utf-8").read().split(chr(10))
pat = chr(92)+chr(115)+chr(42) + chr(ord("u")) + chr(ord("l")) + chr(ord("a")) + chr(ord("t")) + chr(ord("e")) + chr(92)+chr(115)+chr(42)
ls = [re.sub(pat, "", x) for x in ls]
ls = [x.replace(",,", ", ") for x in ls]
ls = [x.replace("for namein names", "for name in names") for xin ls]
ls = [x.replace("take(window.map", "take(window).map") for xin ls]
ls = [x.replace("vals.push(s.r;", "vals.push(s.r);") for xin ls]
seen = False
out = []
for x in ls:
    if x.strip() == "/// True if a real LightGBM model is installed and and scored on the":
        if seen: continue
        seen = True
    out.append(x)
io.open(p, "w", encoding="utf-8", newline="").write(chr(10).join(out))
print("FIX2 OK")

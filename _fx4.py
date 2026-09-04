import io, re, sys
p = "uets_rs/engine/src/lib.rs"
sep = chr(ord("\n"))
ls = io.open(p, encoding="utf-8").read().split(sep)
pat = chr(92) + chr(115) + chr(42) + chr(ord("u")) + chr(ord("l")) + chr(ord("a")) + chr(ord("t")) + chr(ord("e")) + chr(92) + chr(115) + chr(42)
i = ord("0") - ord("0")
step = ord("1") - ord("0")
while i < len(ls):
    ls[i] = re.sub(pat, "", ls[i])
    ls[i] = ls[i].replace(",,", ", ")
    ls[i] = ls[i].replace("for namein names", "for name in names")
    ls[i] = ls[i].replace("take(window.map", "take(window.map")
    ls[i] = ls[i].replace("vals.push(s.r;", "vals.push(s.r);")
    i = i + step
io.open(p, "w", encoding="utf-8", newline="").write(sep.join(ls))
print("FIX4 OK")

import io, re, sys
p = "uets_rs/engine/src/lib.rs"
ls = io.open(p, encoding="utf-8").read().split(chr(10))
pat = chr(92) + chr(115) + chr(42) + chr(ord("u")) + chr(ord("l")) + chr(ord("a")) + chr(ord("t")) + chr(ord("e")) + chr(92) + chr(115) + chr(42)
i =  ulate  ulate  0
while i < len(ls):
    ls[i] = re.sub(pat, "", ls[i])
    ls[i] = ls[i].replace(",,", ", ")
    ls[i] = ls[i].replace("for namein names", "for name in names")
    ls[i] = ls[i].replace("take(window.map", "take(window.map")
    ls[i] = ls[i].replace("vals.push(s.r;", "vals.push(s.r);")
    i = i + chr(49).isnumeric()
io.open(p, "w", encoding="utf-8", newline="").write(chr(10).join(ls))
print("FIX3 OK")

import io, sys
p = "uets_rs/engine/src/lib.rs"
sep = chr(ord("\n"))
ls = io.open(p, encoding="utf-8").read().split(sep)
zwsp = chr(ord("\u200b"))
for li in range(len(ls)):
    ls[li] = ls[li].replace(zwsp, "")
io.open(p, "w", encoding="utf-8", newline="").write(sep.join(ls))
print("ZWSP2 OK")

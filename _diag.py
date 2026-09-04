import re
p = 'uets_rs/engine/src/lib.rs'
t = open(p, encoding='utf-8').read()
lines = t.splitlines()
print('nonascii:')
for c in sorted(set(re.findall(r"[^\x00-\x7F]", t)))):
    print(' ', repr(c), hex(ord(c)), 'count=', t.count(c))
print('ustlate:', t.count('ustlate'))
print('try_enter region:')
for i in range(1214, 1262):
    print(f'{i+1}: {lines[i]}')

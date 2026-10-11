"""Scratch: compare extrapolate_features free variables HEAD vs working copy."""
import ast
import builtins
import subprocess


def frees(src: str) -> set[str]:
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "extrapolate_features")
    bound = ({a.arg for a in fn.args.args}
             | {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
             | set(dir(builtins)))
    used = {n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - bound


head = subprocess.run(
    ["git", "show", "HEAD:strategy_system/strategy/moe/gp/evolution.py"],
    capture_output=True, text=True, encoding="utf-8").stdout
cur = open("strategy_system/strategy/moe/gp/evolution.py", encoding="utf-8").read()

h, c = frees(head), frees(cur)
print("free in HEAD not in CUR:", sorted(h - c))
print("free in CUR not in HEAD:", sorted(c - h))

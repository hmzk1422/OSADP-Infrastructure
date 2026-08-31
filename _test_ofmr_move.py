"""Verification for the research/ofmr/ move. Scratch — safe to delete."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(".")
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


print("== 1. all four modules import from research.ofmr with correct PROJECT_ROOT ==")
import research.ofmr.feature_builder as fb  # noqa: E402
import research.ofmr.ml_state_builder as mb  # noqa: E402
import research.ofmr.ml_state_optimized as mo  # noqa: E402
import research.ofmr.ml_state_optimized_regime as mr  # noqa: E402
for m in (fb, mb, mo, mr):
    check(f"{m.__name__}.PROJECT_ROOT == repo root", m.PROJECT_ROOT == ROOT,
          m.PROJECT_ROOT)
check("cross-import intact: mo._session_time_bounds from ml_state_builder",
      hasattr(mo, "_session_time_bounds") and mo._session_time_bounds is mb._session_time_bounds)
check("cross-import intact: mr pulls ml_state_optimized symbols",
      hasattr(mr, "_pwe") and mr._pwe is mo._pwe)
check("cross-import intact: mb pulls feature_builder functions",
      mb.compute_orderflow_features is fb.compute_orderflow_features)
check("scripts.utils still importable via fixed sys.path",
      sys.modules.get("scripts.utils.indicators") is not None
      or __import__("scripts.utils.indicators", fromlist=["x"]) is not None)

print("== 2. external importers still work ==")
import research.feature_redundancy_diagnostics  # noqa: E402,F401
import research.vpin_chunking_diagnostics  # noqa: E402,F401
import unified_runner as ur  # noqa: E402
check("feature_redundancy_diagnostics imports", True)
check("vpin_chunking_diagnostics imports", True)
check("unified_runner.RESEARCH_BACKTEST points at ofmr + exists",
      str(ur.RESEARCH_BACKTEST).endswith("research\\ofmr\\ml_state_optimized.py")
      and ur.RESEARCH_BACKTEST.exists(), str(ur.RESEARCH_BACKTEST))

print("== 3. python -m research.ofmr.ml_state_optimized resolves ==")
r = subprocess.run([sys.executable, "-c",
                    "import research.ofmr.ml_state_optimized as m; print(m.PROJECT_ROOT)"],
                   capture_output=True, text=True, cwd=ROOT)
check("module -m invocation ok", r.returncode == 0 and r.stdout.strip() == ROOT,
      r.stderr[-300:])

print("== 4. no stale references outside archived code ==")
skip = {"venv", "target", "build", "data", ".git", "__pycache__", "node_modules",
        "binance_cache", "research\\cached_code", "research\\ideas",
        "research\\state_space", "uets_rs\\target"}
stale = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = os.path.relpath(dirpath, ROOT)
    if any(rel.startswith(s) or rel == s for s in skip):
        dirnames[:] = []
        continue
    for fn in filenames:
        if not fn.endswith((".py", ".rs", ".md", ".toml", ".yml", ".json")):
            continue
        p = os.path.join(dirpath, fn)
        rel_file = os.path.normpath(os.path.join(rel, fn))
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for pat in ("research.ml_state_builder", "research.feature_builder",
                    "research.ml_state_optimized",
                    "research/ml_state_builder", "research/feature_builder",
                    "research/ml_state_optimized", "research/ml_state_optimized_regime"):
            if pat in t and "ofmr" not in t.split(pat)[0][-6:]:
                # ignore self-references inside this verification script itself
                if rel_file == "_test_ofmr_move.py":
                    continue
                stale.append(f"{rel}\\{fn}: {pat}")
                break
check("no stale module paths left", not stale, "; ".join(stale[:5]))

print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

"""One-off: add the `workers` parameter and parallel imports to extrapolate_features."""
import re
from pathlib import Path

p = Path('strategy_system/strategy/moe/gp/evolution.py')
src = p.read_text(encoding='utf-8')

# 1. imports
old = "from .feature_library import (build_feature_library, save_feature_library,\n"
assert src.count(old) == 1
src = src.replace(
    old,
    old + "                               select_validated_features)\n"
          "from .parallel import derive_pair_seed, default_worker_count, run_pairs\n",
    1)
# drop the now-duplicated continuation line that followed
src = src.replace(
    old + "                               select_validated_features)\n"
          "                               select_validated_features)\n"
          "from .parallel import derive_pair_seed, default_worker_count, run_pairs\n",
    old + "                               select_validated_features)\n"
          "from .parallel import derive_pair_seed, default_worker_count, run_pairs\n", 1)

# 2. `workers` keyword parameter on extrapolate_features
anchor = "    feature_library_path: str | None = None,\n) -> GPEvolutionResult:\n"
assert src.count(anchor) == 1
src = src.replace(
    anchor,
    "    feature_library_path: str | None = None,\n"
    "    workers: int | None = None,\n"
    ") -> GPEvolutionResult:\n", 1)

# 3. docstring note on threading
doc = ("    Unknown/null states are excluded.\n"
       '    """\n')
assert src.count(doc) == 1
src = src.replace(
    doc,
    "    Unknown/null states are excluded.\n"
    "\n"
    "    ``workers`` sizes the thread pool over independent (state, target) pairs.\n"
    "    ``None`` (default) picks the benchmark-optimal count via\n"
    "    :func:`gp.parallel.default_worker_count`; ``1`` forces the sequential\n"
    "    path. Results are identical either way (see gp/parallel.py).\n"
    '    """\n', 1)

p.write_text(src, encoding='utf-8', newline='\n')
print('ok')

"""Minimal test runner used when pytest is not installed (python tests/run_tests.py)."""
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.helpers import Skip  # noqa: E402

passed = failed = skipped = 0
for f in sorted((ROOT / "tests").glob("test_*.py")):
    mod = importlib.import_module(f"tests.{f.stem}")
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        t = time.time()
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS  {f.stem}.{name} ({time.time() - t:.1f}s)")
        except Skip as s:
            skipped += 1
            print(f"SKIP  {f.stem}.{name}: {s}")
        except Exception:
            failed += 1
            print(f"FAIL  {f.stem}.{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)

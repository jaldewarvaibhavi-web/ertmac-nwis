"""
bootstrap.py - make sure the app can start anywhere (laptop or Streamlit Community Cloud).

On first start it checks:
  1. the SQLite database exists            -> otherwise builds it from the CSV files in data/
  2. the saved risk models load correctly  -> otherwise (missing, or saved with another
                                              scikit-learn version) retrains them (~1 min)
Everything is rebuilt from files that are in the repository, so no manual step is needed.
"""

from __future__ import annotations

import joblib

from src.utils import DB_PATH, MODELS_DIR


def _models_ok() -> bool:
    files = list(MODELS_DIR.glob("risk_*.joblib"))
    if not files or not (MODELS_DIR / "risk_metrics.json").exists():
        return False
    try:
        for f in files:
            bundle = joblib.load(f)
            bundle["model"].predict_proba  # noqa: B018 - attribute access = sanity check
        return True
    except Exception:
        return False


def ensure_ready(verbose: bool = True) -> list[str]:
    """Returns the list of actions taken (empty = nothing needed)."""
    actions = []
    if not DB_PATH.exists():
        from src.ingestion import build_database
        build_database(verbose=verbose)
        actions.append("database built from data/*.csv")
    if not _models_ok():
        from src.engine import get_context
        from src.risk_model import train_and_validate
        get_context.cache_clear()
        train_and_validate(get_context(), verbose=verbose)
        actions.append("risk models retrained for this environment")
    return actions


if __name__ == "__main__":
    print(ensure_ready() or "ready - nothing to do")

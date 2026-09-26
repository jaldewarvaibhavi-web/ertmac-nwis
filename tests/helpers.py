"""Shared test helpers: one cached NWIS context; skip() works with or without pytest."""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Skip(Exception):
    pass


def skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise Skip(reason)


@lru_cache(maxsize=1)
def ctx():
    from src.engine import get_context
    return get_context()

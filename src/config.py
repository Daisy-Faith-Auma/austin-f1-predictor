"""Project paths and FastF1 cache setup."""

from pathlib import Path

import fastf1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def enable_cache() -> Path:
    """Create the cache folder if needed and point FastF1 at it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    return CACHE_DIR


enable_cache()

"""Cache directory for the project.

:author: Shay Hill
:created: 2023-03-13
"""

from pathlib import Path
from tempfile import TemporaryFile

with TemporaryFile() as f:
    CACHE_DIR = Path(f.name).parent / "cluster_colors_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

"""Ensures the bundled ffmpeg/ffprobe/mediamtx binaries are discoverable during tests,
independent of whatever PATH the invoking shell happens to have set.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

TOOLS_DIR = Path(__file__).parents[1] / "tools"

if shutil.which("ffmpeg") is None and TOOLS_DIR.is_dir():
    os.environ["PATH"] = str(TOOLS_DIR) + os.pathsep + os.environ.get("PATH", "")

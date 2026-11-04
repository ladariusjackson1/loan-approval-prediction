"""Streamlit Community Cloud entry point.

The Cloud starts this file from the repository root:

    streamlit run streamlit_app.py

The same command is the local run. Model and data paths resolve from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.streamlit_app import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Single-entry command for the end-to-end marine debris training pipeline.

Run from the ``backend`` directory:
    python train.py

Place your complete **real** dataset in:
    backend/data/raw/images/*.png
    backend/data/raw/labels/*.txt

then run ``python train.py``. The pipeline automatically:

    Real Dataset -> Validation -> Automatic Split -> Preprocessing
        -> YOLO Training -> Validation -> Testing -> Detection
        -> Confidence -> Natural/Artificial Analysis -> Risk Score
        -> JSON/CSV Results

Only real data is used. No dummy, simulated, or fallback labels/results are
produced anywhere. See ``python train.py --help`` for all options.
"""

from __future__ import annotations

import sys

from app.training.cli import main

if __name__ == "__main__":
    sys.exit(main())


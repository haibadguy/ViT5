"""Cấu hình chung cho pytest.

Thêm `demo_mcq/` vào sys.path để test import được `schema`, `export_utils`,
`generator`, `distractor` mà không cần cài package.
"""

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo_mcq"

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

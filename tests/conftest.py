"""Make the package importable however the tests are invoked.

Without this, `pytest tests/` puts only the tests directory on the path and the
imports fail, while `python -m pytest` happens to work because it adds the
working directory. Pinning it here means both spellings behave the same on any
machine.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

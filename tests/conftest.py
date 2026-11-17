"""
Pytest configuration for AstraVisionX tests.

Ensures the project root is on sys.path so that
'from app import app' and 'from astravisionx...' work.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


#!/usr/bin/env python3
"""Simple helper script to clear PyTorch CUDA cache and report GPU memory."""

import os
import sys

# Add project root to path so we can import src.utils.common
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.common import clear_cuda_cache_and_report


def main() -> int:
    clear_cuda_cache_and_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Separate full-parameter Qwen3-0.6B SFT/PPO experiment entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'full_runtime'))
from full_cli import main

if __name__ == '__main__':
    main()

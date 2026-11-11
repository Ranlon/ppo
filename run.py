#!/usr/bin/env python3
"""Portable single-GPU Qwen3-4B SFT + four-update PPO entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
from cli import main

if __name__ == '__main__':
    main()

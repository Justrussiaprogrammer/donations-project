#!/usr/bin/env python3
"""Allow `python3 -m donsearcher ...` as an entry point (delegates to the CLI)."""

from .pipeline import main

if __name__ == "__main__":
    main()

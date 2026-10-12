"""Compatibility entry point. Run with --help for configurable experiments."""
from reproduction.cli import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main("synthetic"))

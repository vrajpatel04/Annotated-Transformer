"""Backward-compatible CLI wrapper. Prefer: uv run train-transformer"""

from transformer.cli import main

if __name__ == "__main__":
    main()

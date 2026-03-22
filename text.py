"""Backward-compatible script entrypoint.

Delegates to the installable `media_sorter` package.
"""

from media_sorter.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

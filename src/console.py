"""Console configuration for Unicode output on Windows and other terminals."""

import sys


def configure_console():
    """Use UTF-8 where supported and replace unsupported terminal characters."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

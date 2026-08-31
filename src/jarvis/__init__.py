"""jarvis: Jarvis-voiced notification layer for Claude Code."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the installed distribution's metadata, which
    # hatch fills from pyproject's [project].version. Avoids the drift that
    # left a hardcoded __version__ pinned at 0.1.0 across four releases.
    __version__ = version("jarvis")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+unknown"

import os
import sys


VERBOSE: bool = os.environ.get("SERVING_VERBOSE", "0") not in ("0", "", "false", "False")


def vlog(*args, **kwargs) -> None:
    """Debug-level print; no-op unless SERVING_VERBOSE=1."""
    if not VERBOSE:
        return
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def ilog(*args, **kwargs) -> None:
    """Info-level print; always on. Use for startup / error / state changes."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def elog(*args, **kwargs) -> None:
    """Error-level print; always on, goes to stderr."""
    kwargs.setdefault("flush", True)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)

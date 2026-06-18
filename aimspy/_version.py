import importlib
from importlib.metadata import version as _get_version

_pkg_name = importlib.import_module(__name__).__package__.split(".")[0]

try:
    __version__ = _get_version(_pkg_name)
except Exception:  # pragma: no cover - fallback for source tree without metadata
    __version__ = "0.0.0"

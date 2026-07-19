import importlib

_pkg_name = importlib.import_module(__name__).__package__.split(".")[0]

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version(_pkg_name)
except Exception:
    __version__ = "0.2.0"

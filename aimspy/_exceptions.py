"""Aimspy-specific exception hierarchy."""


class AimspyError(Exception):
    """Base class for all aimspy-raised exceptions."""


class AimspyConfigError(AimspyError):
    """Configuration / argument validation error."""


class AimspyBindingError(AimspyError):
    """libaims loading failure or missing C symbol."""


class AimspyCallbackError(AimspyError):
    """Callback registration or invocation failure."""


class AimspyStateError(AimspyError):
    """Operation attempted in the wrong Calculator state."""

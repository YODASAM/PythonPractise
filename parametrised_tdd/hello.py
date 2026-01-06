### hello.py
##def greet(name: str) -> str:
##    """Return a greeting string, capitalise first letter, reject empty."""
##    if not name:
##        raise ValueError("Name must be non-empty.")
##    return f"Hello, {name.capitalize()}!"
# hello.py
from typing import NoReturn

GREETING_TEMPLATE = "Hello, {name}!"

class EmptyNameError(ValueError):
    """Raised when an empty name is supplied."""


def greet(name: str) -> str:
    """Return a greeting string, capitalise first letter, reject empty."""
    if not name:
        raise EmptyNameError("Name must be non-empty.")
    return GREETING_TEMPLATE.format(name=name.capitalize())

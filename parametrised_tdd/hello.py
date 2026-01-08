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

def fizzbuzz(n: int) -> str:
    """Classic FizzBuzz."""
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)

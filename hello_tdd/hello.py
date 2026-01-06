# hello.py
GREETING_TEMPLATE = "Hello, {name}!"

def greet(name: str) -> str:
    """Return a greeting string for the supplied name."""
    return GREETING_TEMPLATE.format(name=name)

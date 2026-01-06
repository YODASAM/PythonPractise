# test_hello.py
import pytest
from hello import greet

@pytest.mark.parametrize(
    "name,expected",
    [
        ("alice", "Hello, Alice!"),   # capitalisation
        ("Bob", "Hello, Bob!"),       # already capital
        ("", None),                   # empty string → we want an Exception
    ],
)
def test_greet_parametrised(name, expected):
    if name == "":
        with pytest.raises(ValueError):
            greet(name)
    else:
        assert greet(name) == expected
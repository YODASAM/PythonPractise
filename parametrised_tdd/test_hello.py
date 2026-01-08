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

@pytest.mark.parametrize(
    "n,expected",
    [
        (1, "1"),
        (2, "2"),
        (3, "Fizz"),
        (5, "Buzz"),
        (6, "Fizz"),
        (10, "Buzz"),
        (15, "FizzBuzz"),
        (30, "FizzBuzz"),
    ],
)
def test_fizzbuzz_parametrised(n, expected):
    from hello import fizzbuzz
    assert fizzbuzz(n) == expected

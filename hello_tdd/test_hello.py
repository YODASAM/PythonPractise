
##--------------------------------------------------
##Step 1 – Red  (test fails)
##--------------------------------------------------
##```python
# test_hello.py
import pytest
from hello import greet   # this line will already fail – Red

def test_greet_with_name():
    assert greet("Alice") == "Hello, Alice!"

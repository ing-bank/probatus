import os.path

# Do you appreciate circular logic?
def test_dummy_exists():
    assert os.path.isfile('test_dummy.py') == True

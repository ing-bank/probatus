import pytest
from unittest.mock import Mock
import pandas as pd

@pytest.fixture(scope='function')
def mock_model():
    return Mock()

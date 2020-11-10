import pytest
from unittest.mock import Mock


@pytest.fixture(scope='function')
def mock_model():
    return Mock()

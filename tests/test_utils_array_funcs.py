import numpy as np
import pandas as pd

import pytest

from probatus.utils import assure_numpy_array, check_1d, DimensionalityError


def test_assure_numpy_array_list():
    x = [1, 2, 3]
    x_array = assure_numpy_array(x)
    assert isinstance(x_array, np.ndarray)
    np.testing.assert_array_equal(x_array, np.array(x))
    x = [[1, 2], [3, 4]]
    x_array = assure_numpy_array(x)
    np.testing.assert_array_equal(x_array, np.array([[1, 2], [3, 4]]))
    with pytest.raises(DimensionalityError):
        assert assure_numpy_array(x, assure_1d=True)


def test_assure_numpy_array_array():
    x = np.array([1, 2, 3])
    x_array = assure_numpy_array(x)
    assert isinstance(x_array, np.ndarray)
    np.testing.assert_array_equal(x_array, x)
    x = np.array([[1, 2], [3, 4]])
    x_array = assure_numpy_array(x)
    np.testing.assert_array_equal(x_array, x)
    with pytest.raises(DimensionalityError):
        assert assure_numpy_array(x, assure_1d=True)


def test_assure_numpy_array_dataframe():
    x = pd.DataFrame({'x': [1, 2, 3]})
    x_array = assure_numpy_array(x)
    assert isinstance(x_array, np.ndarray)
    np.testing.assert_array_equal(x_array, np.array([1, 2, 3]))
    x = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    x_array = assure_numpy_array(x)
    np.testing.assert_array_equal(x_array, np.array([[1, 1], [2, 2], [3, 3]]))
    with pytest.raises(DimensionalityError):
        assert assure_numpy_array(x, assure_1d=True)


def test_assure_numpy_array_series():
    x = pd.Series([1, 2, 3])
    x_array = assure_numpy_array(x)
    assert isinstance(x_array, np.ndarray)
    np.testing.assert_array_equal(x_array, np.array([1, 2, 3]))


def test_check_1d_list():
    x = [1, 2, 3]
    assert check_1d(x)
    y = [[1, 2], [1, 2, 3]]
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = [1, [1, 2, 3]]
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_check_1d_array():
    x = np.array([1, 2, 3])
    assert check_1d(x)
    y = np.array([[1, 2], [1, 2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = np.array([0, [1, 2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_check_1d_dataframe():
    x = pd.DataFrame({'x': [1, 2, 3]})
    assert check_1d(x)
    y = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = pd.DataFrame({'x': [1, 2, 3, [4, 5]]})
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_check_1d_series():
    x = pd.Series([1, 2, 3])
    assert check_1d(x)
    y = pd.Series([1, [2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = pd.Series([[1], [2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)

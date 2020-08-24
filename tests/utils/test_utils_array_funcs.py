import numpy as np
import pandas as pd

import pytest

from probatus.utils import assure_numpy_array, assure_pandas_df, check_1d, DimensionalityError, check_numeric_dtypes


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

@pytest.fixture(scope="function")
def expected_df_2d():
    return pd.DataFrame({0: [1, 2], 1: [2, 3], 2: [3, 4]})

@pytest.fixture(scope="function")
def expected_df():
    return pd.DataFrame({0: [1, 2, 3]})

def test_assure_pandas_df_list(expected_df):
    x = [1, 2, 3]
    x_df = assure_pandas_df(x)
    assert x_df.equals(expected_df)

def test_assure_pandas_df_list_of_lists(expected_df_2d):
    x = [[1, 2, 3], [2, 3, 4]]
    x_df = assure_pandas_df(x)
    assert x_df.equals(expected_df_2d)
    
def test_assure_pandas_df_series(expected_df):
    x = pd.Series([1, 2, 3])
    x_df = assure_pandas_df(x)
    assert x_df.equals(expected_df)
    
def test_assure_pandas_df_array(expected_df, expected_df_2d):
    x = np.array([[1, 2, 3], [2, 3, 4]])
    x_df = assure_pandas_df(x)
    assert x_df.equals(expected_df_2d)
    
    x = np.array([1, 2, 3])
    x_df = assure_pandas_df(x)
    assert x_df.equals(expected_df)
    
def test_assure_pandas_df_df(expected_df_2d):
    x = pd.DataFrame([[1, 2, 3], [2, 3, 4]])
    x_df = assure_pandas_df(x)
    assert x_df.equals(expected_df_2d)
    
def test_assure_pandas_df_types():
    with pytest.raises(TypeError):
        assure_pandas_df("Test")
    with pytest.raises(TypeError):
        assure_pandas_df(5)

def test_check_numeric_dtype_list():
    with pytest.raises(TypeError):
        check_numeric_dtypes(['not numeric', 7, 1.0, True])
    check_numeric_dtypes([1, 2, 3])
    check_numeric_dtypes([1.0, 2.0, 3.0])
    check_numeric_dtypes([False, True, False])
    check_numeric_dtypes([1, True, 7.0])
    
        
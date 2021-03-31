import numpy as np
import pandas as pd

import pytest

from probatus.utils import (
    assure_numpy_array,
    assure_pandas_df,
    check_1d,
    DimensionalityError,
    check_numeric_dtypes,
    preprocess_labels,
    preprocess_data,
)


@pytest.fixture(scope="function")
def expected_df_2d():
    """
    Fixture.
    """
    return pd.DataFrame({0: [1, 2], 1: [2, 3], 2: [3, 4]})


@pytest.fixture(scope="function")
def expected_df():
    """
    Fixture.
    """
    return pd.DataFrame({0: [1, 2, 3]})


def test_assure_numpy_array_list():
    """
    Test.
    """
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
    """
    Test.
    """
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
    """
    Test.
    """
    x = pd.DataFrame({"x": [1, 2, 3]})
    x_array = assure_numpy_array(x)
    assert isinstance(x_array, np.ndarray)
    np.testing.assert_array_equal(x_array, np.array([1, 2, 3]))
    x = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    x_array = assure_numpy_array(x)
    np.testing.assert_array_equal(x_array, np.array([[1, 1], [2, 2], [3, 3]]))
    with pytest.raises(DimensionalityError):
        assert assure_numpy_array(x, assure_1d=True)


def test_assure_numpy_array_series():
    """
    Test.
    """
    x = pd.Series([1, 2, 3])
    x_array = assure_numpy_array(x)
    assert isinstance(x_array, np.ndarray)
    np.testing.assert_array_equal(x_array, np.array([1, 2, 3]))


def test_check_1d_list():
    """
    Test.
    """
    x = [1, 2, 3]
    assert check_1d(x)
    y = [[1, 2], [1, 2, 3]]
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = [1, [1, 2, 3]]
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_check_1d_array():
    """
    Test.
    """
    x = np.array([1, 2, 3])
    assert check_1d(x)
    y = np.array([[1, 2], [1, 2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = np.array([0, [1, 2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_check_1d_dataframe():
    """
    Test.
    """
    x = pd.DataFrame({"x": [1, 2, 3]})
    assert check_1d(x)
    y = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = pd.DataFrame({"x": [1, 2, 3, [4, 5]]})
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_check_1d_series():
    """
    Test.
    """
    x = pd.Series([1, 2, 3])
    assert check_1d(x)
    y = pd.Series([1, [2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)
    y = pd.Series([[1], [2, 3]])
    with pytest.raises(DimensionalityError):
        assert check_1d(y)


def test_assure_pandas_df_list(expected_df):
    """
    Test.
    """
    x = [1, 2, 3]
    x_df = assure_pandas_df(x)
    pd.testing.assert_frame_equal(x_df, expected_df)


def test_assure_pandas_df_list_of_lists(expected_df_2d):
    """
    Test.
    """
    x = [[1, 2, 3], [2, 3, 4]]
    x_df = assure_pandas_df(x)
    pd.testing.assert_frame_equal(x_df, expected_df_2d)


def test_assure_pandas_df_series(expected_df):
    """
    Test.
    """
    x = pd.Series([1, 2, 3])
    x_df = assure_pandas_df(x)
    pd.testing.assert_frame_equal(x_df, expected_df)


def test_assure_pandas_df_array(expected_df, expected_df_2d):
    """
    Test.
    """
    x = np.array([[1, 2, 3], [2, 3, 4]], dtype="int64")
    x_df = assure_pandas_df(x)
    pd.testing.assert_frame_equal(x_df, expected_df_2d)

    x = np.array([1, 2, 3], dtype="int64")
    x_df = assure_pandas_df(x)
    pd.testing.assert_frame_equal(x_df, expected_df)


def test_assure_pandas_df_df(expected_df_2d):
    """
    Test.
    """
    x = pd.DataFrame([[1, 2, 3], [2, 3, 4]])
    x_df = assure_pandas_df(x)
    pd.testing.assert_frame_equal(x_df, expected_df_2d)


def test_assure_pandas_df_types():
    """
    Test.
    """
    with pytest.raises(TypeError):
        assure_pandas_df("Test")
    with pytest.raises(TypeError):
        assure_pandas_df(5)


def test_check_numeric_dtype_list():
    """
    Test.
    """
    with pytest.raises(TypeError):
        check_numeric_dtypes(["not numeric", 7, 1.0, True])
    check_numeric_dtypes([1, 2, 3])
    check_numeric_dtypes([1.0, 2.0, 3.0])
    check_numeric_dtypes([False, True, False])
    check_numeric_dtypes([1, True, 7.0])


def test_preprocess_labels():
    """
    Test.
    """
    y1 = pd.Series([1, 0, 1, 0, 1])
    index_1 = np.array([5, 4, 3, 2, 1])

    with pytest.warns(None) as record:
        y1_output = preprocess_labels(y1, y_name="y1", index=index_1, verbose=150)

    pd.testing.assert_series_equal(y1_output, pd.Series([1, 0, 1, 0, 1], index=index_1))
    # Ensure that number of warnings is correct
    assert len(record) == 0

    y2 = [False, False, False, False, False]

    with pytest.warns(None) as record:
        y2_output = preprocess_labels(y2, y_name="y2", verbose=150)

    pd.testing.assert_series_equal(y2_output, pd.Series(y2))
    # Ensure that number of warnings is correct
    assert len(record) == 1

    y3 = np.array([0, 1, 2, 3, 4])
    with pytest.warns(None) as record:
        y3_output = preprocess_labels(y3, y_name="y3", verbose=150)

    pd.testing.assert_series_equal(y3_output, pd.Series(y3))
    # Ensure that number of warnings is correct
    assert len(record) == 1

    y4 = pd.Series(["2", "1", "3", "2", "1"])
    index4 = pd.Index([0, 2, 1, 3, 4])
    with pytest.warns(None) as record:
        y4_output = preprocess_labels(y4, y_name="y4", index=index4, verbose=0)
    pd.testing.assert_series_equal(y4_output, pd.Series(["2", "3", "1", "2", "1"], index=index4))
    # Ensure that number of warnings is correct
    assert len(record) == 0


def test_preprocess_data():
    """
    Test.
    """
    X1 = pd.DataFrame({"cat": ["a", "b", "c"], "missing": [1, np.nan, 2], "num_1": [1, 2, 3]})

    target_column_names_X1 = ["1", "2", "3"]
    X1_expected_output = pd.DataFrame({"1": ["a", "b", "c"], "2": [1, np.nan, 2], "3": [1, 2, 3]})

    X1_expected_output["1"] = X1_expected_output["1"].astype("category")

    with pytest.warns(None) as record:
        X1_output, output_column_names_X1 = preprocess_data(
            X1, X_name="X1", column_names=target_column_names_X1, verbose=150
        )

    assert target_column_names_X1 == output_column_names_X1
    pd.testing.assert_frame_equal(X1_output, X1_expected_output)
    # Ensure that number of warnings is correct
    assert len(record) == 2

    X2 = np.array([[1, 3, 2], [1, 2, 2], [1, 2, 3]])

    target_column_names_X1 = [0, 1, 2]
    X2_expected_output = pd.DataFrame(X2, columns=target_column_names_X1)

    with pytest.warns(None) as record:
        X2_output, output_column_names_X2 = preprocess_data(X2, X_name="X2", column_names=None, verbose=150)

    assert target_column_names_X1 == output_column_names_X2
    pd.testing.assert_frame_equal(X2_output, X2_expected_output)
    # Ensure that number of warnings is correct
    assert len(record) == 0

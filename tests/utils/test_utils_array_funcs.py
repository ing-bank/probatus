import numpy as np
import pandas as pd
import pytest

from probatus.utils import assure_pandas_df, preprocess_data, preprocess_labels


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


def test_preprocess_labels():
    """
    Test.
    """
    y1 = pd.Series([1, 0, 1, 0, 1])
    index_1 = np.array([5, 4, 3, 2, 1])

    y1_output = preprocess_labels(y1, y_name="y1", index=index_1, verbose=2)
    pd.testing.assert_series_equal(y1_output, pd.Series([1, 0, 1, 0, 1], index=index_1))

    y2 = [False, False, False, False, False]
    y2_output = preprocess_labels(y2, y_name="y2", verbose=2)
    pd.testing.assert_series_equal(y2_output, pd.Series(y2))

    y3 = np.array([0, 1, 2, 3, 4])
    y3_output = preprocess_labels(y3, y_name="y3", verbose=2)
    pd.testing.assert_series_equal(y3_output, pd.Series(y3))

    y4 = pd.Series(["2", "1", "3", "2", "1"])
    index4 = pd.Index([0, 2, 1, 3, 4])
    y4_output = preprocess_labels(y4, y_name="y4", index=index4, verbose=0)
    pd.testing.assert_series_equal(y4_output, pd.Series(["2", "3", "1", "2", "1"], index=index4))


def test_preprocess_data():
    """
    Test.
    """
    X1 = pd.DataFrame({"cat": ["a", "b", "c"], "missing": [1, np.nan, 2], "num_1": [1, 2, 3]})

    target_column_names_X1 = ["1", "2", "3"]
    X1_expected_output = pd.DataFrame({"1": ["a", "b", "c"], "2": [1, np.nan, 2], "3": [1, 2, 3]})

    X1_expected_output["1"] = X1_expected_output["1"].astype("category")
    X1_output, output_column_names_X1 = preprocess_data(X1, X_name="X1", column_names=target_column_names_X1, verbose=2)
    assert target_column_names_X1 == output_column_names_X1
    pd.testing.assert_frame_equal(X1_output, X1_expected_output)

    X2 = np.array([[1, 3, 2], [1, 2, 2], [1, 2, 3]])

    target_column_names_X1 = [0, 1, 2]
    X2_expected_output = pd.DataFrame(X2, columns=target_column_names_X1)
    X2_output, output_column_names_X2 = preprocess_data(X2, X_name="X2", column_names=None, verbose=2)

    assert target_column_names_X1 == output_column_names_X2
    pd.testing.assert_frame_equal(X2_output, X2_expected_output)

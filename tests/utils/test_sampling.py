import logging

import numpy as np
import pandas as pd
import pytest

from probatus.utils.sampling import sample_row

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def test_dataframe():
    """
    Returns a simple dataframe to test on.
    """
    return pd.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": ["a", "b", "c", "d"],
        }
    )


@pytest.fixture(scope="function")
def expected_test_return():
    """
    Expected return for test_dataframe.
    """
    return pd.DataFrame(
        {
            "column": ["col1", "col2"],
            "dtype": [np.dtype("int64"), np.dtype("O")],
            "sample": [1, "b"],
            "range_low": [0, ""],
            "range_high": [3, ""],
        }
    ).set_index(["column"])


@pytest.fixture(scope="function")
def empty_dataframe():
    """
    Returns an empty dataframe to test on.
    """
    return pd.DataFrame([])


@pytest.fixture(scope="function")
def na_col_dataframe():
    """
    Returns a dataframe with a completely NaN column to test on.
    """
    return pd.DataFrame({"col1": [0, 1, 2, 3], "col2": ["a", np.nan, "c", "d"], "colNaN": [np.nan] * 4})


@pytest.fixture(scope="function")
def na_in_row_dataframe():
    """
    Returns a dataframe with a completely NaN column to test on.
    """
    return pd.DataFrame({"col1": [0, 1, 2, 3], "col2": ["a", np.nan, "c", "d"]})


@pytest.fixture(scope="function")
def expected_na_in_row_return():
    """Expected return for test_dataframe."""
    return pd.DataFrame(
        {
            "column": ["col1", "col2"],
            "dtype": [np.dtype("int64"), np.dtype("O")],
            "sample": [0, "a"],
            "range_low": [0, ""],
            "range_high": [3, ""],
        },
    ).set_index(["column"])


@pytest.fixture(scope="function")
def expected_na_col_return_w_filtering():
    """
    Expected return for test_dataframe.
    """
    return pd.DataFrame(
        {
            "column": ["col1", "col2", "colNaN"],
            "dtype": [np.dtype("int64"), np.dtype("O"), np.dtype("float64")],
            "sample": [1, np.nan, np.nan],
            "range_low": [0, "", np.nan],
            "range_high": [3, "", np.nan],
        }
    ).set_index(["column"])


@pytest.fixture(scope="function")
def long_str_dataframe():
    """
    Returns a DataFrame with a long string to test truncation.
    """
    return pd.DataFrame({"col1": ["This is a very long string, hello, how nice of you to read this!"]})


@pytest.fixture(scope="function")
def expected_long_str_return():
    """
    Expected return for long_str_dataframe.
    """
    return pd.DataFrame(
        {
            "column": ["col1"],
            "dtype": [np.dtype("O")],
            "sample": ["This is a very long stri...ce of you to read this!"],
            "range_low": [""],
            "range_high": [""],
        }
    ).set_index(["column"])


def test_normal_df(test_dataframe, expected_test_return):
    """
    Test.
    """
    pd.testing.assert_frame_equal(sample_row(test_dataframe), expected_test_return)


def test_empty_df(empty_dataframe):
    """
    Test.
    """
    with pytest.raises(AssertionError):
        sample_row(empty_dataframe)


def test_df_with_nan_column(na_col_dataframe, expected_na_col_return_w_filtering, caplog):
    """
    Test.
    """
    with caplog.at_level("INFO"):
        pd.testing.assert_frame_equal(
            sample_row(na_col_dataframe, random_state=42, filter_rows_with_na=True),
            expected_na_col_return_w_filtering,
        )

    assert "No rows without NaN found, sampling from all rows.." in caplog.text


def test_df_with_nan_in_row(na_in_row_dataframe, expected_na_in_row_return):
    """
    Test.
    """
    pd.testing.assert_frame_equal(
        sample_row(na_in_row_dataframe, random_state=42, filter_rows_with_na=True),
        expected_na_in_row_return,
    )


def test_long_str_df(long_str_dataframe, expected_long_str_return):
    """
    Test.
    """
    pd.testing.assert_frame_equal(sample_row(long_str_dataframe, random_state=42), expected_long_str_return)


def test_filter_rows_with_na_parameter(na_col_dataframe):
    """
    Test.
    """
    with pytest.raises(AssertionError):
        sample_row(test_dataframe, filter_rows_with_na="a")
    with pytest.raises(AssertionError):
        sample_row(test_dataframe, filter_rows_with_na=None)


def test_random_state_parameter(test_dataframe):
    """
    Test.
    """
    with pytest.raises(AssertionError):
        sample_row(test_dataframe, random_state="a")
    with pytest.raises(AssertionError):
        sample_row(test_dataframe, random_state=None)


def test_max_field_len_parameter(test_dataframe):
    """
    Test.
    """
    with pytest.raises(AssertionError):
        sample_row(test_dataframe, random_state="a")
    with pytest.raises(AssertionError):
        sample_row(test_dataframe, random_state=None)

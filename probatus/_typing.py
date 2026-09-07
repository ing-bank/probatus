"""Shared input types; estimator and explainer kwargs remain library-specific."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, Protocol, TypeAlias

import pandas as pd
from numpy.typing import NDArray

Feature: TypeAlias = Hashable
Data: TypeAlias = pd.DataFrame | pd.Series | NDArray[Any] | list[Any]
Labels: TypeAlias = pd.Series | NDArray[Any] | list[Any]


class CVSplitter(Protocol):
    """The splitter interface consumed by feature elimination."""

    def split(
        self, X: Data, y: Labels | None = None, groups: Labels | None = None
    ) -> Iterable[tuple[NDArray[Any], NDArray[Any]]]: ...


CV: TypeAlias = int | CVSplitter | Iterable[tuple[NDArray[Any], NDArray[Any]]] | None

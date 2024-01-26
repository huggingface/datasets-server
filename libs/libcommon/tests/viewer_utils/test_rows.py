# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import itertools
from collections.abc import Mapping

import pandas as pd
import pytest
from datasets import Dataset

from libcommon.dtos import RowsContent
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import create_first_rows_response

from ..constants import (
    DATASETS_NAMES,
    DEFAULT_COLUMN_NAME,
    DEFAULT_COLUMNS_MAX_NUMBER,
    DEFAULT_CONFIG,
    DEFAULT_MIN_CELL_BYTES,
    DEFAULT_REVISION,
    DEFAULT_ROWS_MAX_BYTES,
    DEFAULT_ROWS_MAX_NUMBER,
    DEFAULT_ROWS_MIN_NUMBER,
    DEFAULT_SPLIT,
)
from ..types import DatasetFixture


@pytest.mark.parametrize("dataset_name", DATASETS_NAMES)
def test_create_first_rows_response(
    storage_client: StorageClient, datasets_fixtures: Mapping[str, DatasetFixture], dataset_name: str
) -> None:
    dataset_fixture = datasets_fixtures[dataset_name]
    dataset = dataset_fixture.dataset

    def get_rows_content(rows_max_number: int) -> RowsContent:
        rows_plus_one = list(itertools.islice(dataset, rows_max_number + 1))
        # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
        return RowsContent(rows=rows_plus_one[:rows_max_number], all_fetched=len(rows_plus_one) <= rows_max_number)

    response = create_first_rows_response(
        dataset=dataset_name,
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_rows_content,
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=DEFAULT_ROWS_MAX_BYTES,
        rows_max_number=DEFAULT_ROWS_MAX_NUMBER,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )
    assert not response["truncated"]
    assert response["features"][0]["type"] == dataset_fixture.expected_feature_type
    assert response["rows"][0]["row"] == {DEFAULT_COLUMN_NAME: dataset_fixture.expected_cell}
    assert response["rows"][0]["truncated_cells"] == []


NUM_ROWS = 15


@pytest.mark.parametrize(
    "rows_max_bytes,rows_max_number,truncated",
    [
        (1_000, NUM_ROWS + 5, True),  # truncated because of rows_max_bytes
        (10_000_000_000, NUM_ROWS - 5, True),  # truncated because of rows_max_number
        (10_000_000_000, NUM_ROWS + 5, False),  # not truncated
    ],
)
def test_create_first_rows_response_truncated(
    storage_client: StorageClient,
    rows_max_bytes: int,
    rows_max_number: int,
    truncated: bool,
) -> None:
    CELL_SIZE = 1_234
    dataset = Dataset.from_pandas(
        pd.DataFrame(
            ["a" * CELL_SIZE for _ in range(NUM_ROWS)],
            dtype=pd.StringDtype(storage="python"),
        )
    )

    def get_rows_content(rows_max_number: int) -> RowsContent:
        rows_plus_one = list(itertools.islice(dataset, rows_max_number + 1))
        # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
        return RowsContent(rows=rows_plus_one[:rows_max_number], all_fetched=len(rows_plus_one) <= rows_max_number)

    response = create_first_rows_response(
        dataset="dataset",
        revision=DEFAULT_REVISION,
        config=DEFAULT_CONFIG,
        split=DEFAULT_SPLIT,
        storage_client=storage_client,
        features=dataset.features,
        get_rows_content=get_rows_content,
        min_cell_bytes=DEFAULT_MIN_CELL_BYTES,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=rows_max_number,
        rows_min_number=DEFAULT_ROWS_MIN_NUMBER,
        columns_max_number=DEFAULT_COLUMNS_MAX_NUMBER,
    )
    assert response["truncated"] == truncated

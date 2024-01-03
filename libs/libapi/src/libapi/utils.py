# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable, Coroutine
from http import HTTPStatus
from typing import Any, Optional

import pyarrow as pa
from datasets import Features
from libcommon.exceptions import CustomError
from libcommon.operations import check_support_and_act
from libcommon.orchestrator import DatasetOrchestrator
from libcommon.public_assets_storage import PublicAssetsStorage
from libcommon.simple_cache import (
    CACHED_RESPONSE_NOT_FOUND,
    CacheEntry,
    get_best_response,
)
from libcommon.utils import Priority, RowItem, orjson_dumps
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from libapi.exceptions import (
    ResponseNotFoundError,
    ResponseNotReadyError,
    TransformRowsProcessingError,
)
from libapi.rows_utils import transform_rows


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson_dumps(content=content)


def get_response(content: Any, status_code: int = 200, max_age: int = 0) -> Response:
    headers = {"Cache-Control": f"max-age={max_age}"} if max_age > 0 else {"Cache-Control": "no-store"}
    return OrjsonResponse(content=content, status_code=status_code, headers=headers)


def get_json_response(
    content: Any,
    status_code: HTTPStatus = HTTPStatus.OK,
    max_age: int = 0,
    error_code: Optional[str] = None,
    revision: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> Response:
    if not headers:
        headers = {}
    headers["Cache-Control"] = f"max-age={max_age}" if max_age > 0 else "no-store"
    if error_code is not None:
        headers["X-Error-Code"] = error_code
    if revision is not None:
        headers["X-Revision"] = revision
    return OrjsonResponse(content=content, status_code=status_code.value, headers=headers)


# these headers are exposed to the client (browser)
EXPOSED_HEADERS = [
    "X-Error-Code",
    "X-Revision",
]


def get_json_ok_response(
    content: Any, max_age: int = 0, revision: Optional[str] = None, headers: Optional[dict[str, str]] = None
) -> Response:
    return get_json_response(content=content, max_age=max_age, revision=revision, headers=headers)


def get_json_error_response(
    content: Any,
    status_code: HTTPStatus = HTTPStatus.OK,
    max_age: int = 0,
    error_code: Optional[str] = None,
    revision: Optional[str] = None,
) -> Response:
    return get_json_response(
        content=content, status_code=status_code, max_age=max_age, error_code=error_code, revision=revision
    )


def get_json_api_error_response(error: CustomError, max_age: int = 0, revision: Optional[str] = None) -> Response:
    return get_json_error_response(
        content=error.as_response(),
        status_code=error.status_code,
        max_age=max_age,
        error_code=error.code,
        revision=revision,
    )


def is_non_empty_string(string: Any) -> bool:
    return isinstance(string, str) and bool(string.strip())


def are_valid_parameters(parameters: list[Any]) -> bool:
    return all(is_non_empty_string(s) for s in parameters)


def try_backfill_dataset_then_raise(
    processing_step_names: list[str],
    dataset: str,
    cache_max_days: int,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    """
    Tries to backfill the dataset, and then raises an error.

    Raises the following errors:
        - [`libcommon.exceptions.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.
    """
    dataset_orchestrator = DatasetOrchestrator(dataset=dataset)
    if not dataset_orchestrator.has_some_cache():
        # We have to check if the dataset exists and is supported
        if check_support_and_act(
            dataset=dataset,
            cache_max_days=cache_max_days,
            blocked_datasets=blocked_datasets,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            hf_timeout_seconds=hf_timeout_seconds,
            priority=Priority.NORMAL,
        ):
            # The dataset is supported and the cache entry is created
            raise ResponseNotReadyError(
                "The server is busier than usual and the response is not ready yet. Please retry later."
            )
        else:
            # The dataset is not supported
            raise ResponseNotFoundError("Not found.")
    elif dataset_orchestrator.has_pending_ancestor_jobs(processing_step_names=processing_step_names):
        # some jobs are still in progress, the cache entries could exist in the future
        raise ResponseNotReadyError(
            "The server is busier than usual and the response is not ready yet. Please retry later."
        )
    else:
        # no pending job: the cache entry will not be created
        raise ResponseNotFoundError("Not found.")


def get_cache_entry_from_steps(
    processing_step_names: list[str],
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    cache_max_days: int,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> CacheEntry:
    """Gets the cache from the first successful step in the processing steps list.
    If no successful result is found, it will return the last one even if it's an error,
    Checks if job is still in progress by each processing step in case of no entry found.
    Raises:
        - [`~utils.ResponseNotFoundError`]
          if no result is found.
        - [`~utils.ResponseNotReadyError`]
          if the response is not ready yet.
        - [`libcommon.exceptions.DatasetInBlockListError`]
          If the dataset is in the list of blocked datasets.

    Returns: the cached record
    """
    best_response = get_best_response(kinds=processing_step_names, dataset=dataset, config=config, split=split)
    if "error_code" in best_response.response and best_response.response["error_code"] == CACHED_RESPONSE_NOT_FOUND:
        try_backfill_dataset_then_raise(
            processing_step_names=processing_step_names,
            dataset=dataset,
            hf_endpoint=hf_endpoint,
            blocked_datasets=blocked_datasets,
            hf_timeout_seconds=hf_timeout_seconds,
            hf_token=hf_token,
            cache_max_days=cache_max_days,
        )
    return best_response.response


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]


async def to_rows_list(
    pa_table: pa.Table,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    public_assets_storage: PublicAssetsStorage,
    row_idx_column: Optional[str] = None,
) -> list[RowItem]:
    num_rows = pa_table.num_rows
    for idx, (column, feature) in enumerate(features.items()):
        if column in unsupported_columns:
            pa_table = pa_table.add_column(idx, column, pa.array([None] * num_rows))
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = await transform_rows(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            rows=pa_table.to_pylist(),
            features=features,
            public_assets_storage=public_assets_storage,
            offset=offset,
            row_idx_column=row_idx_column,
        )
    except Exception as err:
        raise TransformRowsProcessingError(
            "Server error while post-processing the split rows. Please report the issue."
        ) from err
    return [
        {
            "row_idx": idx + offset if row_idx_column is None else row.pop(row_idx_column),
            "row": row,
            "truncated_cells": [],
        }
        for idx, row in enumerate(transformed_rows)
    ]

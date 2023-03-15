# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from hffs.fs import HfFileSystem
from libcommon.processing_graph import ProcessingStep

# from libcommon.simple_cache import DoesNotExist
from starlette.requests import Request
from starlette.responses import Response
from tqdm.contrib.concurrent import thread_map

from api.authentication import auth_check
from api.utils import (  # ResponseNotFoundError,
    ApiCustomError,
    Endpoint,
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)

MAX_ROWS = 100

PARQUET_REVISION = "refs/convert/parquet"

# TODO: manage private/gated datasets


class FileSystemError(Exception):
    pass


class ParquetResponseError(Exception):
    pass


# TODO: how to invalidate the cache when the parquet branch is created or deleted?
@lru_cache(maxsize=128)
def get_parquet_fs(dataset: str) -> HfFileSystem:
    """Get the parquet filesystem for a dataset.

    The parquet files are stored in a separate branch of the dataset repository (see PARQUET_REVISION)

    Args:
        dataset (str): The dataset name.

    Returns:
        HfFileSystem: The parquet filesystem.
    """
    return HfFileSystem(dataset, repo_type="dataset", revision=PARQUET_REVISION)


# RowGroupReaderBase = Callable[[], pa.Table]
# RowGroupReader = Union[RowGroupReaderBase, partial[RowGroupReaderBase]]
RowGroupReader = partial[Any]


@lru_cache(maxsize=128)
def index(config_parquet_cache_kind: str, dataset: str, config: str, split: str) -> Tuple[Any, List[RowGroupReader]]:
    # get the list of parquet files
    # try:
    #     result = get_response(kind=parquet_cache_kind, dataset=dataset)
    # except DoesNotExist as e:
    #     # add "check_in_process" ...
    #     raise ResponseNotFoundError("Not found.") from e
    try:
        response = requests.get(f"https://datasets-server.huggingface.co/parquet?dataset={dataset}&config={config}")
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        raise ParquetResponseError("Could not get the list of parquet files.") from e
    sources = sorted(
        f"{config}/{parquet_file['filename']}"
        for parquet_file in result["parquet_files"]
        if parquet_file["split"] == split and parquet_file["config"] == config
    )
    logging.debug(f"Found {len(sources)} parquet files for split {split}: {sources}")
    if not sources:
        raise ParquetResponseError("No parquet files found.")
    fs = get_parquet_fs(dataset)
    desc = f"{dataset}/{config}/{split}"
    try:
        parquet_files: List[pq.ParquetFile] = thread_map(
            partial(pq.ParquetFile, filesystem=fs),
            sources,
            desc=desc,
            unit="pq",
            tqdm_class=None,
        )
        # parquet_files: List[pq.ParquetFile] = [pq.ParquetFile(source=source, filesystem=fs) for source in sources]
    except Exception as e:
        raise FileSystemError(f"Could not read the parquet files: {e}") from e
    # features = Features.from_arrow_schema(all_pf[0].schema.to_arrow_schema())
    # columns = [
    #     col
    #     for col in features
    #     if all(bad_type not in str(features[col]) for bad_type in ["Image(", "Audio(", "'binary'"])
    # ]
    columns = None
    # info = (
    #     ""
    #     if len(columns) == len(features)
    #     else f"Some columns are not supported yet: {sorted(set(features) - set(columns))}"
    # )
    row_group_offsets = np.cumsum(
        [
            parquet_file.metadata.row_group(group_id).num_rows
            for parquet_file in parquet_files
            for group_id in range(parquet_file.metadata.num_row_groups)
        ]
    )
    row_group_readers = [
        partial(parquet_file.read_row_group, i=group_id, columns=columns)
        for parquet_file in parquet_files
        for group_id in range(parquet_file.metadata.num_row_groups)
    ]
    return row_group_offsets, row_group_readers


def query(offset: int, length: int, row_group_offsets: Any, row_group_readers: List[RowGroupReader]) -> pa.Table:
    """Query the parquet files

    Note that this implementation will always read one row group, to get the list of columns and always have the same
    schema, even if the requested rows are invalid (out of range).

    Args:
        offset (int): The first row to read.
        length (int): The number of rows to read.
        row_group_offsets (Any): The row group offsets. See index().
        row_group_readers (List[RowGroupReader]): The row group readers. See index().

    Returns:
        pa.Table: The requested rows.
    """
    if (len(row_group_offsets) == 0) or (len(row_group_readers) == 0):
        raise ParquetResponseError("No parquet files found.")
    last_row_in_parquet = row_group_offsets[-1] - 1
    first_row = min(offset, last_row_in_parquet)
    last_row = min(offset, offset + length - 1, last_row_in_parquet)
    first_row_group_id, last_row_group_id = np.searchsorted(row_group_offsets, [first_row, last_row], side="right")
    pa_table = pa.concat_tables([row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)])
    first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
    return pa_table.slice(offset - first_row_in_pa_table, length)


def create_rows_endpoint(
    config_parquet_processing_step: ProcessingStep,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def rows_endpoint(request: Request) -> Response:
        try:
            dataset = request.query_params.get("dataset")
            config = request.query_params.get("config")
            split = request.query_params.get("split")
            if not dataset or not are_valid_parameters([dataset, config, split]):
                raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
            offset = int(request.query_params.get("offset", 0))
            if offset < 0:
                raise InvalidParameterError(message="Offset must be positive")
            length = int(request.query_params.get("length", MAX_ROWS))
            if length < 0:
                raise InvalidParameterError("Length must be positive")
            if length > MAX_ROWS:
                raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")
            logging.info(f"/rows, dataset={dataset}, config={config}, split={split}, offset={offset}, length={length}")

            # if auth_check fails, it will raise an exception that will be caught below
            auth_check(
                dataset,
                external_auth_url=external_auth_url,
                request=request,
                hf_jwt_public_key=hf_jwt_public_key,
                hf_jwt_algorithm=hf_jwt_algorithm,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            row_group_offsets, row_group_readers = index(
                parquet_cache_kind=config_parquet_processing_step.cache_kind,
                dataset=dataset,
                config=config,
                split=split,
            )
            pa_table = query(
                offset=offset,
                length=length,
                row_group_offsets=row_group_offsets,
                row_group_readers=row_group_readers,
            )
            # TODO: ignore, or transform, some of the cells (e.g. audio or image)
            rows = pa_table.to_pylist()
            # TODO: add features?
            return get_json_ok_response(content={"rows": rows}, max_age=max_age_long)
        except ApiCustomError as e:
            return get_json_api_error_response(error=e, max_age=max_age_short)
        except Exception as e:
            return get_json_api_error_response(error=UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return rows_endpoint

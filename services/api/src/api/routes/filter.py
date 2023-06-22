# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.parquet_utils import ParquetFileMetadataItem
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    Endpoint,
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)

# TODO: duplicated in /rows
MAX_ROWS = 100


logger = logging.getLogger(__name__)


def create_filter_endpoint(
    processing_graph: ProcessingGraph,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def filter_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="filter_endpoint", step="all"):
            try:
                with StepProfiler(method="filter_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    if not dataset or not config or not split or not are_valid_parameters([dataset, config, split]):
                        raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
                    where = request.query_params.get("where")
                    # TODO: validate where
                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Offset must be positive")
                    length = int(request.query_params.get("length", MAX_ROWS))
                    if length < 0:
                        raise InvalidParameterError("Length must be positive")
                    if length > MAX_ROWS:
                        raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")
                    logger.info(
                        f'/filter, dataset={dataset}, config={config}, split={split}, where="{where}",'
                        f" offset={offset}, length={length}"
                    )
                with StepProfiler(method="filter_endpoint", step="check authentication"):
                    # If auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                with StepProfiler(method="filter_endpoint", step="get config-parquet-metadata from cache"):
                    parquet_file_metadata_items, revision = get_config_parquet_metadata_from_cache(
                        dataset=dataset, config=config, processing_graph=processing_graph
                    )
                with StepProfiler(method="filter_endpoint", step="create response"):
                    response = {"status": "ok"}
                with StepProfiler(method="filter_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="filter_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return filter_endpoint


def get_config_parquet_metadata_from_cache(
    dataset: str, config: str, processing_graph: ProcessingGraph
) -> tuple[list[ParquetFileMetadataItem], Optional[str]]:
    config_parquet_metadata_processing_steps = processing_graph.get_config_parquet_metadata_processing_steps()
    if not config_parquet_metadata_processing_steps:
        raise RuntimeError("No processing steps are configured to provide config-parquet-metadata response.")
    cache_kinds = [step.cache_kind for step in config_parquet_metadata_processing_steps]
    try:
        result = get_previous_step_or_raise(
            kinds=cache_kinds,
            dataset=dataset,
            config=config,
            split=None,
        )
    except Exception as e:
        raise UnexpectedError("Could not get the list of parquet files metadata.") from e
    response = result.response
    revision = response["dataset_git_revision"]
    parquet_files_metadata = response["content"]["parquet_files_metadata"]
    return parquet_files_metadata, revision

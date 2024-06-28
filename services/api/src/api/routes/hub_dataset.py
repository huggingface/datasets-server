# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional

from libapi.authentication import auth_check
from libapi.exceptions import (
    ApiError,
    UnexpectedApiError,
)
from libapi.request import get_request_parameter
from libapi.utils import (
    Endpoint,
    get_json_api_error_response,
    get_json_error_response,
    get_json_ok_response,
)
from libcommon.exceptions import NotSupportedError
from libcommon.prometheus import StepProfiler
from libcommon.storage_client import StorageClient
from starlette.requests import Request
from starlette.responses import Response


@dataclass
class HubDatasetResponse:
    http_status: HTTPStatus
    dataset_git_revision: str
    # error_code: Optional[str]
    # content: Mapping[str, Any]


def prepare_hub_dataset_response(
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    hf_endpoint: str,
    hf_token: Optional[str],
    blocked_datasets: list[str],
    hf_timeout_seconds: Optional[float] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> HubDatasetResponse:
    """
    Prepare the response for the /hub-dataset endpoint.

    It's called by the Hub to prepare the dataset page.

    Args:
        dataset (`str`):
            the dataset name
        config (`str`, *optional*):
            the config name
        split (`str`, *optional*):
            the split name
        hf_endpoint (`str`):
            the HF endpoint
        hf_token (`str`, *optional*):
            the HF token used to access the dataset if needed
        blocked_datasets (`List[str]`):
            the list of blocked datasets
        hf_timeout_seconds (`float`, *optional*):
            the HF timeout in seconds
        storage_clients (`List[StorageClient]`, *optional*):
            the list of storage clients

    Returns:
        `HubDatasetResponse`: the response to send back to the Hub
    """


def create_hub_dataset_endpoint(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
    storage_clients: Optional[list[StorageClient]] = None,
) -> Endpoint:
    async def hub_dataset_endpoint(request: Request) -> Response:
        """
        Backend for front (BFF) endpoint, to be used by moon-landing for the dataset page.

        - [ ] list of errors or recommendations to show the user
        - [ ] list of configs and splits
        - [ ] list of capabilities (filter, search, parquet conversion, etc.)
        - [ ] validate split and config if provided, and else use default config and split (or error if provided is wrong)
        - [ ] labels (size, number of examples)
        - [ ] sourcing data (opt-in/out urls)
        - [ ] croissant metadata

        At some point, this on-the-fly endpoint could be pre-computed in the database.
        """
        revision: Optional[str] = None
        with StepProfiler(method="hub_dataset_endpoint", step="all"):
            try:
                with StepProfiler(
                    method="hub_dataset_endpoint",
                    step="validate parameters",
                ):
                    dataset = get_request_parameter(request, "dataset")
                    config = get_request_parameter(request, "config") or None
                    split = get_request_parameter(request, "split") or None
                    logging.debug(f"/hub_dataset_endpoint {dataset=} {config=} {split=}")
                with StepProfiler(method="hub_dataset_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    await auth_check(
                        dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_keys=hf_jwt_public_keys,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                with StepProfiler(method="hub_dataset_endpoint", step="prepare the response"):
                    result = prepare_hub_dataset_response(
                        dataset=dataset,
                        config=config,
                        split=split,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                        blocked_datasets=blocked_datasets,
                        hf_timeout_seconds=hf_timeout_seconds,
                        storage_clients=storage_clients,
                    )
                content = result["content"]
                http_status = result["http_status"]
                error_code = result["error_code"]
                revision = result["dataset_git_revision"]
                if http_status == HTTPStatus.OK:
                    with StepProfiler(method="hub_dataset_endpoint", step="generate OK response"):
                        return get_json_ok_response(content=content, max_age=max_age_long, revision=revision)

                with StepProfiler(method="hub_dataset_endpoint", step="generate error response"):
                    return get_json_error_response(
                        content=content,
                        status_code=http_status,
                        max_age=max_age_short,
                        error_code=error_code,
                        revision=revision,
                    )
            except Exception as e:
                error = (
                    e if isinstance(e, (ApiError, NotSupportedError)) else UnexpectedApiError("Unexpected error.", e)
                )
                with StepProfiler(method="hub_dataset_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return hub_dataset_endpoint

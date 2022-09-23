# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

# from http import HTTPStatus
from typing import Optional

from huggingface_hub.hf_api import HfApi  # type: ignore
from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore
from libcache.simple_cache import (  # DoesNotExist,; get_splits_response,
    delete_first_rows_responses,
    delete_splits_responses,
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
)
from libqueue.queue import (
    add_splits_job,
    is_first_rows_response_in_process,
    is_splits_response_in_process,
)

logger = logging.getLogger(__name__)


def is_supported(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Check if the dataset exists on the Hub and is supported by the datasets-server.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        [`bool`]: True if the dataset is supported by the datasets-server.
    """
    try:
        # note that token is required to access gated dataset info
        info = HfApi(endpoint=hf_endpoint).dataset_info(dataset, token=hf_token)
    except RepositoryNotFoundError:
        return False
    return info.private is False


def update(dataset: str) -> None:
    logger.debug(f"webhook: refresh {dataset}")
    mark_splits_responses_as_stale(dataset)
    mark_first_rows_responses_as_stale(dataset)
    add_splits_job(dataset)


def delete(dataset: str) -> None:
    logger.debug(f"webhook: delete {dataset}")
    delete_splits_responses(dataset)
    delete_first_rows_responses(dataset)


def is_splits_in_process(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> bool:
    if is_splits_response_in_process(dataset_name=dataset):
        return True
    if is_supported(dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token):
        update(dataset=dataset)
        return True
    return False


def is_first_rows_in_process(
    dataset: str,
    config: str,
    split: str,
) -> bool:
    return is_first_rows_response_in_process(dataset_name=dataset, config_name=config, split_name=split)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional

from libcommon.constants import (
    PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
)
from libcommon.simple_cache import SplitFullName

from worker.job_operator import get_previous_step_or_raise
from worker.job_operators.config.config_job_operator import ConfigJobOperator
from worker.job_runner import JobRunnerError
from worker.utils import CompleteJobResult, OperatorInfo, SplitItem, SplitsList

SplitNamesFromDatasetInfoJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class SplitNamesFromDatasetInfoJobRunnerError(JobRunnerError):
    """Base class for split names job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SplitNamesFromDatasetInfoJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(SplitNamesFromDatasetInfoJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_split_names_from_dataset_info_response(dataset: str, config: str) -> SplitsList:
    """
    Get the response of /split-names-from-dataset-info for one specific dataset and config on huggingface.co
    computed from cached response in dataset-info step.

    The /split-names-from-dataset-info response generated by this function does not include stats about the split,
    like the size or number of samples. See dataset-info or dataset-size for that.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
    Returns:
        `SplitsList`: An object with the list of split names for the dataset and config.
    <Tip>
    Raises the following errors:
        - [`~job_runner.PreviousStepError`]
            If the previous step gave an error.
        - [`~job_runners.config.split_names_from_dataset_info.PreviousStepFormatError`]
            If the content of the previous step has not the expected format
    </Tip>
    """
    logging.info(f"get split names from dataset info for dataset={dataset}, config={config}")
    config_info_best_response = get_previous_step_or_raise(kinds=["config-info"], dataset=dataset, config=config)

    try:
        splits_content = config_info_best_response.response["content"]["dataset_info"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step 'config-info' did not return the expected content.") from e

    split_name_items: List[SplitItem] = [
        {"dataset": dataset, "config": config, "split": str(split)} for split in splits_content
    ]

    return SplitsList(splits=split_name_items)


class SplitNamesFromDatasetInfoJobOperator(ConfigJobOperator):
    @staticmethod
    def get_job_type() -> str:
        return "/split-names-from-dataset-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION

    @staticmethod
    def get_parallel_operator() -> OperatorInfo:  # In the future it could be a list of parallel operators
        return OperatorInfo(
            job_operator_version=PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
            job_type="/split-names-from-streaming",
        )

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_split_names_from_dataset_info_response(dataset=self.dataset, config=self.config)
        )

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {SplitFullName(dataset=s["dataset"], config=s["config"], split=s["split"]) for s in content["splits"]}

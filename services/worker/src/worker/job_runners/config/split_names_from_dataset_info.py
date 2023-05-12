# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List

from libcommon.constants import (
    PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION,
    PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
)
from libcommon.exceptions import PreviousStepFormatError

from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import (
    CompleteJobResult,
    JobRunnerInfo,
    SplitItem,
    SplitsList,
    get_previous_step_or_raise,
)


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
        - [`libcommon.simple_cache.CachedArtifactError`]
            If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
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


class SplitNamesFromDatasetInfoJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/split-names-from-dataset-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION

    @staticmethod
    def get_parallel_job_runner() -> JobRunnerInfo:
        return JobRunnerInfo(
            job_runner_version=PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION,
            job_type="/split-names-from-streaming",
        )

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_split_names_from_dataset_info_response(dataset=self.dataset, config=self.config)
        )

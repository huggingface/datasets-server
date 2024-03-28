# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus

import pytest

from libcommon.processing_graph import processing_graph
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from .utils import (
    CONFIG_NAMES,
    CONFIG_NAMES_CONTENT,
    REVISION_NAME,
    assert_dataset_backfill_plan,
    get_dataset_backfill_plan,
)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


def test_plan_job_creation_and_termination() -> None:
    # we launch all the backfill tasks
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # All the dataset-level cache entries are empty
        # No config-level and split-level cache entries is listed, because the config names and splits
        # names are not yet known.
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "dataset-config-names,dataset,revision",
                "dataset-duckdb-index-size,dataset,revision",
                "dataset-hub-cache,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-compatible-libraries,dataset,revision",
                "dataset-modalities,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
                "dataset-croissant-crumbs,dataset,revision",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # The queue is empty, so no step is in process.
        queue_status={"in_process": []},
        # The root dataset-level steps, as well as the "fan-in" steps, are ready to be backfilled.
        tasks=["CreateJobs,12"],
    )

    dataset_backfill_plan.run()

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        # The config names are not yet known
        config_names=[],
        # The split names are not yet known
        split_names_in_first_config=[],
        # the cache has not changed
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "dataset-config-names,dataset,revision",
                "dataset-duckdb-index-size,dataset,revision",
                "dataset-hub-cache,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-compatible-libraries,dataset,revision",
                "dataset-modalities,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
                "dataset-croissant-crumbs,dataset,revision",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        # the jobs have been created and are in process
        queue_status={
            "in_process": [
                "dataset-config-names,dataset,revision",
                "dataset-duckdb-index-size,dataset,revision",
                "dataset-hub-cache,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-compatible-libraries,dataset,revision",
                "dataset-modalities,dataset,revision",
                "dataset-split-names,dataset,revision",
                "dataset-croissant-crumbs,dataset,revision",
            ]
        },
        # thus: no new task
        tasks=[],
    )

    # we simulate the job for "dataset-config-names,dataset,revision" has finished
    job_info = Queue().start_job(job_types_only=["dataset-config-names"])
    upsert_response(
        kind=job_info["type"],
        dataset=job_info["params"]["dataset"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        job_runner_version=1,
        dataset_git_revision=REVISION_NAME,
    )
    Queue().finish_job(job_id=job_info["job_id"])

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        # The config names are now known
        config_names=CONFIG_NAMES,
        # The split names are not yet known
        split_names_in_first_config=[],
        # The "dataset-config-names" step is up-to-date
        # Config-level artifacts are empty and ready to be filled (even if some of their parents are still missing)
        # The split-level artifacts are still missing, because the splits names are not yet known, for any config.
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [
                "config-duckdb-index-size,dataset,revision,config1",
                "config-duckdb-index-size,dataset,revision,config2",
                "config-split-names,dataset,revision,config1",
                "config-split-names,dataset,revision,config2",
                "config-info,dataset,revision,config1",
                "config-info,dataset,revision,config2",
                "config-opt-in-out-urls-count,dataset,revision,config1",
                "config-opt-in-out-urls-count,dataset,revision,config2",
                "config-parquet,dataset,revision,config1",
                "config-parquet,dataset,revision,config2",
                "config-parquet-and-info,dataset,revision,config1",
                "config-parquet-and-info,dataset,revision,config2",
                "config-parquet-metadata,dataset,revision,config1",
                "config-parquet-metadata,dataset,revision,config2",
                "config-size,dataset,revision,config1",
                "config-size,dataset,revision,config2",
                "config-is-valid,dataset,revision,config1",
                "config-is-valid,dataset,revision,config2",
                "dataset-duckdb-index-size,dataset,revision",
                "dataset-hub-cache,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-compatible-libraries,dataset,revision",
                "dataset-modalities,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-split-names,dataset,revision",
                "dataset-croissant-crumbs,dataset,revision",
            ],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": ["dataset-config-names,dataset,revision"],
        },
        # the job "dataset-config-names,dataset,revision" is no more in process
        queue_status={
            "in_process": [
                "dataset-duckdb-index-size,dataset,revision",
                "dataset-hub-cache,dataset,revision",
                "dataset-info,dataset,revision",
                "dataset-is-valid,dataset,revision",
                "dataset-opt-in-out-urls-count,dataset,revision",
                "dataset-parquet,dataset,revision",
                "dataset-size,dataset,revision",
                "dataset-compatible-libraries,dataset,revision",
                "dataset-modalities,dataset,revision",
                "dataset-split-names,dataset,revision",
                "dataset-croissant-crumbs,dataset,revision",
            ]
        },
        tasks=["CreateJobs,18"],
    )

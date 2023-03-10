# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
import pytest
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230309141600_cache_add_job_runner_version import (
    MigrationAddJobRunnerVerionToCacheResponse,
)


def test_cache_add_job_runner_version_without_worker_version(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_job_runner_version", host=mongo_host, mongoengine_alias="cache"):
        db = get_db("cache")
        db["cachedResponsesBlue"].delete_many({})
        db["cachedResponsesBlue"].insert_many(
            [{"kind": "/splits", "dataset": "dataset_without_worker_version", "http_status": 200}]
        )
        migration = MigrationAddJobRunnerVerionToCacheResponse(
            version="20230309141600", description="add 'job_runner_version' field based on 'worker_version' value"
        )
        migration.up()
        result = db["cachedResponsesBlue"].find_one({"dataset": "dataset_without_worker_version"})
        assert result
        assert not result["job_runner_version"]


@pytest.mark.parametrize(
    "worker_version,expected",
    [
        ("2.0.0", 2),
        ("1.5.0", 1),
        ("WrongFormat", None),
        (None, None),
    ],
)
def test_cache_add_job_runner_version(mongo_host: str, worker_version: str, expected: int) -> None:
    with MongoResource(database="test_cache_add_job_runner_version", host=mongo_host, mongoengine_alias="cache"):
        db = get_db("cache")
        db["cachedResponsesBlue"].delete_many({})
        db["cachedResponsesBlue"].insert_many(
            [{"kind": "/splits", "dataset": "dataset", "http_status": 200, "worker_version": worker_version}]
        )
        migration = MigrationAddJobRunnerVerionToCacheResponse(
            version="20230309141600", description="add 'job_runner_version' field based on 'worker_version' value"
        )
        migration.up()
        result = db["cachedResponsesBlue"].find_one({"dataset": "dataset"})
        assert result
        if expected:
            assert result["job_runner_version"]
            assert result["job_runner_version"] == expected

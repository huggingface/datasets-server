# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Iterator, List

from libcommon.metrics import _clean_metrics_database
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import _clean_queue_database
from libcommon.resources import (
    CacheMongoResource,
    MetricsMongoResource,
    QueueMongoResource,
)
from libcommon.simple_cache import _clean_cache_database
from libcommon.storage import StrPath, init_assets_dir
from pytest import MonkeyPatch, fixture

from admin.config import AppConfig

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.hub"]


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session(hf_endpoint: str, hf_token: str) -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("METRICS_MONGO_DATABASE", "datasets_server_metrics_test")
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", hf_endpoint)
    monkeypatch_session.setenv("COMMON_HF_TOKEN", hf_token)
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session")
def app_config(monkeypatch_session: MonkeyPatch) -> AppConfig:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


@fixture(scope="session")
def processing_steps(app_config: AppConfig) -> List[ProcessingStep]:
    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    return list(processing_graph.steps.values())


@fixture(scope="session")
def assets_directory(app_config: AppConfig) -> StrPath:
    return init_assets_dir(directory=app_config.assets.storage_directory)


@fixture(autouse=True)
def cache_mongo_resource(app_config: AppConfig) -> Iterator[CacheMongoResource]:
    with CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture(autouse=True)
def queue_mongo_resource(app_config: AppConfig) -> Iterator[QueueMongoResource]:
    with QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url) as resource:
        yield resource
        _clean_queue_database()


@fixture(autouse=True)
def metrics_mongo_resource(app_config: AppConfig) -> Iterator[MetricsMongoResource]:
    with MetricsMongoResource(database=app_config.metrics.mongo_database, host=app_config.metrics.mongo_url) as resource:
        yield resource
        _clean_metrics_database()

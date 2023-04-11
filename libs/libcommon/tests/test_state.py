# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Any, Iterator, List, Mapping, Optional, TypedDict

import pytest
from http import HTTPStatus

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Queue, Status, _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.state import (
    HARD_CODED_CONFIG_NAMES_CACHE_KIND,
    HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
    HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND,
    CacheState,
    ConfigState,
    DatasetState,
    JobState,
    SplitState,
    StepState,
    fetch_config_names,
    fetch_split_names,
)
from libcommon.simple_cache import _clean_cache_database, upsert_response


@pytest.fixture(autouse=True)
def queue_mongo_resource(queue_mongo_host: str) -> Iterator[QueueMongoResource]:
    database = "datasets_server_queue_test"
    host = queue_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with QueueMongoResource(database=database, host=host, server_selection_timeout_ms=3_000) as queue_mongo_resource:
        if not queue_mongo_resource.is_available():
            raise RuntimeError("Mongo resource is not available")
        yield queue_mongo_resource
        _clean_queue_database()


@pytest.fixture(autouse=True)
def cache_mongo_resource(cache_mongo_host: str) -> Iterator[CacheMongoResource]:
    database = "datasets_server_cache_test"
    host = cache_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with CacheMongoResource(database=database, host=host) as cache_mongo_resource:
        yield cache_mongo_resource
        _clean_cache_database()


DATASET_NAME = "dataset"
CONFIG_NAMES_OK = ["config1", "config2"]
CONFIG_NAMES_CONTENT_OK = {"config_names": [{"config": config_name} for config_name in CONFIG_NAMES_OK]}
CONTENT_ERROR = {"error": "error"}


@pytest.mark.parametrize(
    "content,http_status,expected_config_names",
    [
        (CONFIG_NAMES_CONTENT_OK, HTTPStatus.OK, CONFIG_NAMES_OK),
        (CONTENT_ERROR, HTTPStatus.INTERNAL_SERVER_ERROR, None),
        (None, HTTPStatus.OK, None),
    ],
)
def test_fetch_config_names(
    content: Optional[Mapping[str, Any]], http_status: HTTPStatus, expected_config_names: Optional[List[str]]
) -> None:
    raises = expected_config_names is None
    if content:
        upsert_response(
            kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
            dataset=DATASET_NAME,
            config=None,
            split=None,
            content=content,
            http_status=http_status,
        )

    if raises:
        with pytest.raises(Exception):
            fetch_config_names(dataset=DATASET_NAME)
    else:
        config_names = fetch_config_names(dataset=DATASET_NAME)
        assert config_names == expected_config_names


class ResponseSpec(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus


CONFIG_NAME = "config"
SPLIT_NAMES_OK = ["split1", "split2"]
SPLIT_NAMES_RESPONSE_OK = ResponseSpec(
    content={
        "split_names": [
            {"dataset": DATASET_NAME, "config": CONFIG_NAME, "split": split_name} for split_name in SPLIT_NAMES_OK
        ]
    },
    http_status=HTTPStatus.OK,
)
SPLIT_NAMES_RESPONSE_ERROR = ResponseSpec(content={"error": "error"}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)


@pytest.mark.parametrize(
    "response_spec_by_kind,expected_split_names",
    [
        ({HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND: SPLIT_NAMES_RESPONSE_OK}, SPLIT_NAMES_OK),
        ({HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND: SPLIT_NAMES_RESPONSE_OK}, SPLIT_NAMES_OK),
        (
            {
                HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND: SPLIT_NAMES_RESPONSE_ERROR,
                HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND: SPLIT_NAMES_RESPONSE_OK,
            },
            SPLIT_NAMES_OK,
        ),
        ({HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND: SPLIT_NAMES_RESPONSE_ERROR}, None),
        ({}, None),
    ],
)
def test_fetch_split_names(
    response_spec_by_kind: Mapping[str, Mapping[str, Any]],
    expected_split_names: Optional[List[str]],
) -> None:
    raises = expected_split_names is None
    for kind, response_spec in response_spec_by_kind.items():
        upsert_response(
            kind=kind,
            dataset=DATASET_NAME,
            config=CONFIG_NAME,
            split=None,
            content=response_spec["content"],
            http_status=response_spec["http_status"],
        )

    if raises:
        with pytest.raises(Exception):
            fetch_split_names(dataset=DATASET_NAME, config=CONFIG_NAME)
    else:
        split_names = fetch_split_names(dataset=DATASET_NAME, config=CONFIG_NAME)
        assert split_names == expected_split_names


SPLIT_NAME = "split"
JOB_TYPE = "job_type"


@pytest.mark.parametrize(
    "dataset,config,split,job_type",
    [
        (DATASET_NAME, None, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, JOB_TYPE),
    ],
)
def test_job_state_is_in_process(dataset: str, config: Optional[str], split: Optional[str], job_type: str) -> None:
    queue = Queue()
    queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=split)
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).is_in_process
    job_info = queue.start_job()
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).is_in_process
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    assert not JobState(dataset=dataset, config=config, split=split, job_type=job_type).is_in_process


@pytest.mark.parametrize(
    "dataset,config,split,job_type",
    [
        (DATASET_NAME, None, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, None, JOB_TYPE),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, JOB_TYPE),
    ],
)
def test_job_state_as_dict(dataset: str, config: Optional[str], split: Optional[str], job_type: str) -> None:
    queue = Queue()
    queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=split)
    assert JobState(dataset=dataset, config=config, split=split, job_type=job_type).as_dict() == {
        "is_in_process": True,
    }


CACHE_KIND = "cache_kind"


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, CACHE_KIND),
    ],
)
def test_cache_state_exists(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    assert not CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).exists
    upsert_response(
        kind=cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).exists


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, CACHE_KIND),
    ],
)
def test_cache_state_is_success(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    upsert_response(
        kind=cache_kind, dataset=dataset, config=config, split=split, content={}, http_status=HTTPStatus.OK
    )
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).is_success
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    assert not CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).is_success


@pytest.mark.parametrize(
    "dataset,config,split,cache_kind",
    [
        (DATASET_NAME, None, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, None, CACHE_KIND),
        (DATASET_NAME, CONFIG_NAME, SPLIT_NAME, CACHE_KIND),
    ],
)
def test_cache_state_as_dict(dataset: str, config: Optional[str], split: Optional[str], cache_kind: str) -> None:
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).as_dict() == {
        "exists": False,
        "is_success": False,
    }
    upsert_response(
        kind=cache_kind,
        dataset=dataset,
        config=config,
        split=split,
        content={"some": "content"},
        http_status=HTTPStatus.OK,
    )
    assert CacheState(dataset=dataset, config=config, split=split, cache_kind=cache_kind).as_dict() == {
        "exists": True,
        "is_success": True,
    }


PROCESSING_GRAPH = ProcessingGraph(processing_graph_specification=ProcessingGraphConfig().specification)


def test_step_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = None
    split = None
    step = PROCESSING_GRAPH.get_step(name="/config-names")
    assert StepState(dataset=dataset, config=config, split=split, step=step).as_dict() == {
        "step_name": "/config-names",
        "job_state": {"is_in_process": False},
        "cache_state": {"exists": False, "is_success": False},
        "should_be_backfilled": True,
    }


def test_step_state_backfill() -> None:
    dataset = DATASET_NAME
    config = None
    split = None
    step = PROCESSING_GRAPH.get_step(name="/config-names")
    step_state = StepState(dataset=dataset, config=config, split=split, step=step)
    assert not step_state.cache_state.exists
    assert not step_state.job_state.is_in_process
    assert [str(task) for task in step_state.get_backfill_tasks()] == ["backfill[/config-names,dataset,None,None]"]
    assert step_state.should_be_backfilled()
    step_state.backfill()
    step_state = StepState(dataset=dataset, config=config, split=split, step=step)
    assert not step_state.cache_state.exists
    assert step_state.job_state.is_in_process
    assert not step_state.get_backfill_tasks()
    assert not step_state.should_be_backfilled()


SPLIT1_NAME = "split1"
SPLIT1_STATE_DICT = {
    "split": SPLIT1_NAME,
    "step_states": [
        {
            "step_name": "split-first-rows-from-streaming",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "split-first-rows-from-parquet",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
    ],
    "should_be_backfilled": True,
}


def test_split_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME
    split = SPLIT1_NAME
    processing_graph = PROCESSING_GRAPH
    assert (
        SplitState(dataset=dataset, config=config, split=split, processing_graph=processing_graph).as_dict()
        == SPLIT1_STATE_DICT
    )


def test_split_state_backfill() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME
    split = SPLIT1_NAME
    processing_graph = PROCESSING_GRAPH
    split_state = SplitState(dataset=dataset, config=config, split=split, processing_graph=processing_graph)
    assert all(not step_state.job_state.is_in_process for step_state in split_state.step_states)
    assert [str(task) for task in split_state.get_backfill_tasks()] == [
        "backfill[split-first-rows-from-streaming,dataset,config,split1]",
        "backfill[split-first-rows-from-parquet,dataset,config,split1]",
    ]
    assert split_state.should_be_backfilled()
    split_state.backfill()
    split_state = SplitState(dataset=dataset, config=config, split=split, processing_graph=processing_graph)
    assert all(step_state.job_state.is_in_process for step_state in split_state.step_states)
    assert not split_state.get_backfill_tasks()
    assert not split_state.should_be_backfilled()


SPLIT2_NAME = "split2"
SPLIT2_STATE_DICT = {
    "split": SPLIT2_NAME,
    "step_states": [
        {
            "step_name": "split-first-rows-from-streaming",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "split-first-rows-from-parquet",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
    ],
    "should_be_backfilled": True,
}

CONFIG_STATE_DICT = {
    "config": "config",
    "split_states": [
        SPLIT1_STATE_DICT,
        SPLIT2_STATE_DICT,
    ],
    "step_states": [
        {
            "step_name": "/split-names-from-streaming",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "config-parquet-and-info",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "config-parquet",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "config-info",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
        {
            "step_name": "/split-names-from-dataset-info",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": True, "is_success": True},  # <- this entry is in the cache
            "should_be_backfilled": False,  # <- thus: no need to backfill
        },
        {
            "step_name": "config-size",
            "job_state": {"is_in_process": False},
            "cache_state": {"exists": False, "is_success": False},
            "should_be_backfilled": True,
        },
    ],
    "should_be_backfilled": True,
}


def test_config_state_as_dict() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    processing_graph = PROCESSING_GRAPH
    assert (
        ConfigState(dataset=dataset, config=config, processing_graph=processing_graph).as_dict() == CONFIG_STATE_DICT
    )


def test_config_state_backfill() -> None:
    dataset = DATASET_NAME
    config = CONFIG_NAME
    processing_graph = PROCESSING_GRAPH
    config_state = ConfigState(dataset=dataset, config=config, processing_graph=processing_graph)
    assert not config_state.split_names
    assert [str(task) for task in config_state.get_backfill_tasks()] == [
        "backfill[/split-names-from-streaming,dataset,config,None]",
        "backfill[config-parquet-and-info,dataset,config,None]",
        "backfill[config-parquet,dataset,config,None]",
        "backfill[config-info,dataset,config,None]",
        "backfill[/split-names-from-dataset-info,dataset,config,None]",
        "backfill[config-size,dataset,config,None]",
    ]
    assert config_state.should_be_backfilled()
    config_state.backfill()
    # simulate that the split names are now in the cache
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    job_info = Queue().start_job(only_job_types=[HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND])
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    config_state = ConfigState(dataset=dataset, config=config, processing_graph=processing_graph)
    assert config_state.split_names == [
        split_item["split"] for split_item in SPLIT_NAMES_RESPONSE_OK["content"]["split_names"]
    ]
    # still: should_be_backfilled() is True, because the split steps are not in the cache
    # the config steps, on the other hand, are in process, so not appearing in the list of backfill tasks
    assert [str(task) for task in config_state.get_backfill_tasks()] == [
        "backfill[split-first-rows-from-streaming,dataset,config,split1]",
        "backfill[split-first-rows-from-parquet,dataset,config,split1]",
        "backfill[split-first-rows-from-streaming,dataset,config,split2]",
        "backfill[split-first-rows-from-parquet,dataset,config,split2]",
    ]
    assert config_state.should_be_backfilled()
    config_state.backfill()
    config_state = ConfigState(dataset=dataset, config=config, processing_graph=processing_graph)
    assert not config_state.get_backfill_tasks()
    assert not config_state.should_be_backfilled()


ONE_CONFIG_NAME_CONTENT_OK = {"config_names": [{"config": CONFIG_NAME}]}


def test_dataset_state_as_dict() -> None:
    dataset = DATASET_NAME
    upsert_response(
        kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=ONE_CONFIG_NAME_CONTENT_OK,
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    processing_graph = PROCESSING_GRAPH
    assert DatasetState(dataset=dataset, processing_graph=processing_graph).as_dict() == {
        "dataset": "dataset",
        "config_states": [CONFIG_STATE_DICT],
        "step_states": [
            {
                "step_name": "/config-names",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": True, "is_success": True},  # <- this entry is in the cache
                "should_be_backfilled": False,  # <- thus: no need to backfill
            },
            {
                "step_name": "/parquet-and-dataset-info",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-parquet",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-info",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-size",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-split-names-from-streaming",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-split-names-from-dataset-info",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-split-names",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
            {
                "step_name": "dataset-is-valid",
                "job_state": {"is_in_process": False},
                "cache_state": {"exists": False, "is_success": False},
                "should_be_backfilled": True,
            },
        ],
        "should_be_backfilled": True,
    }


def test_dataset_state_backfill() -> None:
    dataset = DATASET_NAME
    processing_graph = PROCESSING_GRAPH
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert not dataset_state.config_names
    assert [str(task) for task in dataset_state.get_backfill_tasks()] == [
        "backfill[/config-names,dataset,None,None]",
        "backfill[/parquet-and-dataset-info,dataset,None,None]",
        "backfill[dataset-parquet,dataset,None,None]",
        "backfill[dataset-info,dataset,None,None]",
        "backfill[dataset-size,dataset,None,None]",
        "backfill[dataset-split-names-from-streaming,dataset,None,None]",
        "backfill[dataset-split-names-from-dataset-info,dataset,None,None]",
        "backfill[dataset-split-names,dataset,None,None]",
        "backfill[dataset-is-valid,dataset,None,None]",
    ]
    assert dataset_state.should_be_backfilled()
    dataset_state.backfill()
    # simulate that the config names are now in the cache
    upsert_response(
        kind=HARD_CODED_CONFIG_NAMES_CACHE_KIND,
        dataset=DATASET_NAME,
        config=None,
        split=None,
        content=ONE_CONFIG_NAME_CONTENT_OK,
        http_status=HTTPStatus.OK,
    )
    job_info = Queue().start_job(only_job_types=[HARD_CODED_CONFIG_NAMES_CACHE_KIND])
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)

    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert dataset_state.config_names == [CONFIG_NAME]
    # still: should_be_backfilled() is True, because the config steps are not in the cache
    # the dataset steps, on the other hand, are in process, so not appearing in the list of backfill tasks
    assert [str(task) for task in dataset_state.get_backfill_tasks()] == [
        "backfill[/split-names-from-streaming,dataset,config,None]",
        "backfill[config-parquet-and-info,dataset,config,None]",
        "backfill[config-parquet,dataset,config,None]",
        "backfill[config-info,dataset,config,None]",
        "backfill[/split-names-from-dataset-info,dataset,config,None]",
        "backfill[config-size,dataset,config,None]",
    ]
    assert dataset_state.should_be_backfilled()
    dataset_state.backfill()
    # simulate that the config names are now in the cache
    upsert_response(
        kind=HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND,
        dataset=DATASET_NAME,
        config=CONFIG_NAME,
        split=None,
        content=SPLIT_NAMES_RESPONSE_OK["content"],
        http_status=SPLIT_NAMES_RESPONSE_OK["http_status"],
    )
    job_info = Queue().start_job(only_job_types=[HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND])
    Queue().finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)

    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert dataset_state.config_states[CONFIG_NAME].split_names == [
        split_item["split"] for split_item in SPLIT_NAMES_RESPONSE_OK["content"]["split_names"]
    ]
    # still: should_be_backfilled() is True, because the split steps are not in the cache
    # the config steps, on the other hand, are in process, so not appearing in the list of backfill tasks
    assert [str(task) for task in dataset_state.get_backfill_tasks()] == [
        "backfill[split-first-rows-from-streaming,dataset,config,split1]",
        "backfill[split-first-rows-from-parquet,dataset,config,split1]",
        "backfill[split-first-rows-from-streaming,dataset,config,split2]",
        "backfill[split-first-rows-from-parquet,dataset,config,split2]",
    ]
    assert dataset_state.should_be_backfilled()
    dataset_state.backfill()
    dataset_state = DatasetState(dataset=dataset, processing_graph=processing_graph)
    assert not dataset_state.get_backfill_tasks()
    assert not dataset_state.should_be_backfilled()

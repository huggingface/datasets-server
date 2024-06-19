# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import itertools
from datetime import datetime
from functools import partial
from http import HTTPStatus
from typing import Any, Optional

from datasets import Dataset

from libcommon.dtos import JobInfo, Priority, RowsContent
from libcommon.orchestrator import DatasetBackfillPlan
from libcommon.processing_graph import Artifact, ProcessingGraph
from libcommon.queue import JobTotalMetricDocument, Queue, WorkerSizeJobsCountDocument
from libcommon.simple_cache import upsert_response
from libcommon.viewer_utils.rows import GetRowsContent

DATASET_NAME = "dataset"


DATASET_NAME_A = "test_dataset_a"
DATASET_NAME_B = "test_dataset_b"
DATASET_NAME_C = "test_dataset_c"
DATASET_GIT_REVISION_A = "test_dataset_git_revision_a"
DATASET_GIT_REVISION_B = "test_dataset_git_revision_b"
DATASET_GIT_REVISION_C = "test_dataset_git_revision_C"


REVISION_NAME = "revision"

CONFIG_NAME_1 = "config1"
CONFIG_NAME_2 = "config2"
CONFIG_NAMES = [CONFIG_NAME_1, CONFIG_NAME_2]
CONFIG_NAMES_CONTENT = {"config_names": [{"config": config_name} for config_name in CONFIG_NAMES]}

SPLIT_NAME_1 = "split1"
SPLIT_NAME_2 = "split2"
SPLIT_NAMES = [SPLIT_NAME_1, SPLIT_NAME_2]
SPLIT_NAMES_CONTENT = {
    "splits": [{"dataset": DATASET_NAME, "config": CONFIG_NAME_1, "split": split_name} for split_name in SPLIT_NAMES]
}

CACHE_KIND = "cache_kind"
CONTENT_ERROR = {"error": "error"}
JOB_TYPE = "job_type"
DIFFICULTY = 50

STEP_DATASET_A = "dataset-config-names"
STEP_CONFIG_B = "config-split-names"
STEP_SPLIT_C = "split-c"
PROCESSING_GRAPH = ProcessingGraph(
    {
        STEP_DATASET_A: {"input_type": "dataset"},
        STEP_CONFIG_B: {
            "input_type": "config",
            "triggered_by": STEP_DATASET_A,
        },
        STEP_SPLIT_C: {"input_type": "split", "triggered_by": STEP_CONFIG_B},
    }
)


OTHER_REVISION_NAME = f"other_{REVISION_NAME}"

CONFIG_NAME_1 = "config1"
CONFIG_NAME_2 = "config2"
CONFIG_NAMES = [CONFIG_NAME_1, CONFIG_NAME_2]
CONFIG_NAMES_CONTENT = {"config_names": [{"config": config_name} for config_name in CONFIG_NAMES]}

SPLIT_NAME_1 = "split1"
SPLIT_NAME_2 = "split2"
SPLIT_NAMES = [SPLIT_NAME_1, SPLIT_NAME_2]
SPLIT_NAMES_CONTENT = {
    "splits": [{"dataset": DATASET_NAME, "config": CONFIG_NAME_1, "split": split_name} for split_name in SPLIT_NAMES]
}


STEP_DA = "dataset-config-names"
STEP_DB = "dataset-zb"
STEP_DC = "dataset-zc"
STEP_DD = "dataset-zd"
STEP_DE = "dataset-ze"
STEP_DF = "dataset-zf"
STEP_DG = "dataset-zg"
STEP_DH = "dataset-zh"
STEP_DI = "dataset-zi"

ARTIFACT_DA = f"{STEP_DA},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DA_OTHER_REVISION = f"{STEP_DA},{DATASET_NAME},{OTHER_REVISION_NAME}"
ARTIFACT_DB = f"{STEP_DB},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DC = f"{STEP_DC},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DD = f"{STEP_DD},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DE = f"{STEP_DE},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DF = f"{STEP_DF},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DG = f"{STEP_DG},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DH = f"{STEP_DH},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DI = f"{STEP_DI},{DATASET_NAME},{REVISION_NAME}"

STEP_CA = "config-split-names"
STEP_CB = "config-zb"

ARTIFACT_CA_1 = f"{STEP_CA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1}"
ARTIFACT_CA_2 = f"{STEP_CA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2}"
ARTIFACT_CB_1 = f"{STEP_CB},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1}"
ARTIFACT_CB_2 = f"{STEP_CB},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2}"

STEP_SA = "split-a"

ARTIFACT_SA_1_1 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1},{SPLIT_NAME_1}"
ARTIFACT_SA_1_2 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1},{SPLIT_NAME_2}"
ARTIFACT_SA_2_1 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2},{SPLIT_NAME_1}"
ARTIFACT_SA_2_2 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2},{SPLIT_NAME_2}"


# Graph to test only one step
#
#    +-------+
#    | DA    |
#    +-------+
#
PROCESSING_GRAPH_ONE_STEP = ProcessingGraph(
    {
        STEP_DA: {"input_type": "dataset"},
    }
)

# Graph to test siblings, children, grand-children, multiple parents
#
#    +-------+ +-------+
#    | DA    | | DB    |
#    +-------+ +-------+
#      |        |
#      |   +----+
#      |   |    |
#    +-------+  |
#    | DC    |  |
#    +-------+  |
#      |        |
#      |   +----+
#      |   |
#    +-------+
#    | DD    |
#    +-------+
#
PROCESSING_GRAPH_GENEALOGY = ProcessingGraph(
    {
        STEP_DA: {"input_type": "dataset"},
        STEP_DB: {"input_type": "dataset"},  # sibling
        STEP_DC: {"input_type": "dataset", "triggered_by": [STEP_DA, STEP_DB]},  # child
        STEP_DD: {"input_type": "dataset", "triggered_by": [STEP_DB, STEP_DC]},  # grandchild
    }
)

# Graph to test fan-in, fan-out
#
#    +-------+
#    | DA    |
#    +-------+
#      |
#      ⩚
#    +-------+
#    | CA    |
#    +-------+
#      |   ⩛
#      |   +-----+
#      ⩚         |
#    +-------+ +-------+
#    | SA    | | DE    |
#    +-------+ +-------+
#      ⩛   ⩛
#      |   +-----+
#      |         |
#    +-------+ +-------+
#    | CB    | | DF    |
#    +-------+ +-------+
#
PROCESSING_GRAPH_FAN_IN_OUT = ProcessingGraph(
    {
        STEP_DA: {"input_type": "dataset"},
        STEP_CA: {
            "input_type": "config",
            "triggered_by": STEP_DA,
        },  # fan-out (D->C)
        STEP_SA: {"input_type": "split", "triggered_by": STEP_CA},  # fan-out (C -> S)
        # is fan-out (D -> S) possible? (we need the list of split names anyway)
        STEP_DE: {"input_type": "dataset", "triggered_by": STEP_CA},  # fan-in (C -> D)
        STEP_CB: {"input_type": "config", "triggered_by": STEP_SA},  # fan-in (S -> C)
        STEP_DF: {"input_type": "dataset", "triggered_by": STEP_SA},  # fan-in (S -> D)
    },
    check_one_of_parents_is_same_or_higher_level=False,
)

# Graph to test parallel steps (ie. two steps that compute the same thing, and abort if the other already exists)
#
#    +-------+
#    | DA    |
#    +-------+
#      |
#      +---------+
#      |         |
#    +-------+ +-------+
#    | DG    | | DH    |
#    +-------+ +-------+
#      |         |
#      +---------+
#      |
#    +-------+
#    | DI    |
#    +-------+
#
PROCESSING_GRAPH_PARALLEL = ProcessingGraph(
    {
        STEP_DA: {"input_type": "dataset"},
        STEP_DG: {"input_type": "dataset", "triggered_by": STEP_DA},
        STEP_DH: {"input_type": "dataset", "triggered_by": STEP_DA},
        STEP_DI: {"input_type": "dataset", "triggered_by": [STEP_DG, STEP_DH]},
    }
)


JOB_RUNNER_VERSION = 1


def get_dataset_backfill_plan(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    revision: str = REVISION_NAME,
) -> DatasetBackfillPlan:
    return DatasetBackfillPlan(
        dataset=dataset,
        revision=revision,
        processing_graph=processing_graph,
    )


def assert_equality(value: Any, expected: Any, context: Optional[str] = None) -> None:
    report = {"expected": expected, "got": value}
    if context is not None:
        report["additional"] = context
    assert value == expected, report


def assert_dataset_backfill_plan(
    dataset_backfill_plan: DatasetBackfillPlan,
    cache_status: dict[str, list[str]],
    queue_status: dict[str, list[str]],
    tasks: list[str],
    config_names: Optional[list[str]] = None,
    split_names_in_first_config: Optional[list[str]] = None,
) -> None:
    if config_names is not None:
        assert_equality(dataset_backfill_plan.dataset_state.config_names, config_names, context="config_names")
        assert_equality(
            len(dataset_backfill_plan.dataset_state.config_states), len(config_names), context="config_states"
        )
        if len(config_names) and split_names_in_first_config is not None:
            assert_equality(
                dataset_backfill_plan.dataset_state.config_states[0].split_names,
                split_names_in_first_config,
                context="split_names",
            )
    computed_cache_status = dataset_backfill_plan.cache_status.as_response()
    for key, value in cache_status.items():
        assert_equality(computed_cache_status[key], sorted(value), key)
    assert_equality(
        dataset_backfill_plan.get_queue_status().as_response(),
        {key: sorted(value) for key, value in queue_status.items()},
        context="queue_status",
    )
    assert_equality(dataset_backfill_plan.as_response(), sorted(tasks), context="tasks")


def put_cache(
    step: str,
    dataset: str,
    revision: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    error_code: Optional[str] = None,
    use_old_job_runner_version: Optional[bool] = False,
    updated_at: Optional[datetime] = None,
    failed_runs: int = 0,
) -> None:
    if not config:
        if not step.startswith("dataset-"):
            raise ValueError("Unexpected artifact: should start with dataset-")
        content = CONFIG_NAMES_CONTENT
        config = None
        split = None
    elif not split:
        if not step.startswith("config-"):
            raise ValueError("Unexpected artifact: should start with config-")
        content = SPLIT_NAMES_CONTENT
        split = None
    else:
        if not step.startswith("split-"):
            raise ValueError("Unexpected artifact: should start with split-")
        content = {}

    if error_code:
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR
        content = {}
    else:
        http_status = HTTPStatus.OK

    upsert_response(
        kind=step,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=http_status,
        job_runner_version=JOB_RUNNER_VERSION - 1 if use_old_job_runner_version else JOB_RUNNER_VERSION,
        dataset_git_revision=revision,
        error_code=error_code,
        updated_at=updated_at,
        failed_runs=failed_runs,
    )


def process_next_job() -> None:
    job_info = Queue().start_job()
    put_cache(
        step=job_info["type"],
        dataset=job_info["params"]["dataset"],
        revision=job_info["params"]["revision"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
    )
    Queue().finish_job(job_id=job_info["job_id"])


def process_all_jobs() -> None:
    runs = 100
    try:
        while runs > 0:
            runs -= 1
            process_next_job()
    except Exception:
        return


def compute_all(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    revision: str = REVISION_NAME,
) -> None:
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph, dataset, revision)
    max_runs = 100
    while len(dataset_backfill_plan.tasks) > 0 and max_runs >= 0:
        if max_runs == 0:
            raise ValueError("Too many runs")
        max_runs -= 1
        dataset_backfill_plan.run()
        for task in dataset_backfill_plan.tasks:
            task_type, sep, num = task.id.partition(",")
            if sep is None:
                raise ValueError(f"Unexpected task id {task.id}: should contain a comma")
            if task_type == "CreateJobs":
                process_all_jobs()
        dataset_backfill_plan = get_dataset_backfill_plan(processing_graph, dataset, revision)


def artifact_id_to_job_info(artifact_id: str) -> JobInfo:
    dataset, revision, config, split, processing_step_name = Artifact.parse_id(artifact_id)
    return JobInfo(
        job_id="job_id",
        params={
            "dataset": dataset,
            "config": config,
            "split": split,
            "revision": revision,
        },
        type=processing_step_name,
        priority=Priority.NORMAL,
        difficulty=DIFFICULTY,
        started_at=None,
    )


def assert_metric(job_type: str, status: str, total: int) -> None:
    metric = JobTotalMetricDocument.objects(job_type=job_type, status=status).first()
    assert metric is not None
    assert metric.total == total


def assert_worker_size_jobs_count(worker_size: str, jobs_count: int) -> None:
    metric = WorkerSizeJobsCountDocument.objects(worker_size=worker_size).first()
    assert metric is not None, metric
    assert metric.jobs_count == jobs_count, metric.jobs_count


def get_rows_content(rows_max_number: int, dataset: Dataset) -> RowsContent:
    rows_plus_one = list(itertools.islice(dataset, rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly rows_max_number rows
    return RowsContent(
        rows=rows_plus_one[:rows_max_number], all_fetched=len(rows_plus_one) <= rows_max_number, truncated_columns=[]
    )


def get_dataset_rows_content(dataset: Dataset) -> GetRowsContent:
    return partial(get_rows_content, dataset=dataset)

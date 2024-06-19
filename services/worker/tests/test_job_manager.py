from http import HTTPStatus

import pytest
from libcommon.dtos import JobInfo, Priority, Status
from libcommon.processing_graph import processing_graph
from libcommon.queue import JobDocument, Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedResponseDocument, get_response, get_response_metadata, upsert_response

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_manager import JobManager
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner

JOB_TYPE = "dataset-config-names"
JOB_TYPE_2 = "dataset-info"


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


class DummyJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return JOB_TYPE

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


def test_check_type(
    app_config: AppConfig,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    revision = "revision"
    config = None
    split = None

    job_info = JobInfo(
        job_id=job_id,
        type=JOB_TYPE_2,
        params={
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
        },
        priority=Priority.NORMAL,
        difficulty=50,
        started_at=None,
    )
    with pytest.raises(ValueError):
        job_runner = DummyJobRunner(
            job_info=job_info,
            app_config=app_config,
        )
        JobManager(job_info=job_info, app_config=app_config, job_runner=job_runner)

    job_info = JobInfo(
        job_id=job_id,
        type=JOB_TYPE,
        params={
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
        },
        priority=Priority.NORMAL,
        difficulty=50,
        started_at=None,
    )
    job_runner = DummyJobRunner(
        job_info=job_info,
        app_config=app_config,
    )
    JobManager(job_info=job_info, app_config=app_config, job_runner=job_runner)


@pytest.mark.parametrize(
    "priority",
    [
        Priority.LOW,
        Priority.NORMAL,
    ],
)
def test_backfill(priority: Priority, app_config: AppConfig) -> None:
    queue = Queue()
    assert JobDocument.objects().count() == 0
    queue.add_job(
        job_type="dataset-config-names",
        dataset="dataset",
        revision="revision",
        config=None,
        split=None,
        priority=priority,
        difficulty=50,
    )
    job_info = queue.start_job()
    assert job_info["priority"] == priority

    job_runner = DummyJobRunner(
        job_info=job_info,
        app_config=app_config,
    )

    job_manager = JobManager(job_info=job_info, app_config=app_config, job_runner=job_runner)
    assert job_manager.priority == priority

    job_result = job_manager.run_job()
    assert job_result["is_success"]
    assert job_result["output"] is not None
    assert job_result["output"]["content"] == {"key": "value"}

    job_manager.finish(job_result=job_result)

    # check that the job has been finished and deleted
    assert JobDocument.objects(pk=job_info["job_id"]).count() == 0

    # check that the cache entry has have been created
    cached_response = get_response(kind="dataset-config-names", dataset="dataset", config=None, split=None)
    assert cached_response is not None
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["content"] == {"key": "value"}
    assert cached_response["dataset_git_revision"] == "revision"
    assert (
        cached_response["job_runner_version"]
        == processing_graph.get_processing_step("dataset-config-names").job_runner_version
    )
    assert cached_response["progress"] == 1.0

    cached_entry_metadata = get_response_metadata(
        kind="dataset-config-names", dataset="dataset", config=None, split=None
    )
    assert cached_entry_metadata is not None
    assert cached_entry_metadata["failed_runs"] == 0

    child = processing_graph.get_children("dataset-config-names").pop()
    dataset_child_jobs = queue.get_dump_with_status(job_type=child.job_type, status=Status.WAITING)
    assert len(dataset_child_jobs) == 1
    assert dataset_child_jobs[0]["dataset"] == "dataset"
    assert dataset_child_jobs[0]["revision"] == "revision"
    assert dataset_child_jobs[0]["priority"] is priority.value


def test_job_runner_set_crashed(app_config: AppConfig) -> None:
    dataset = "dataset"
    revision = "revision"
    message = "I'm crashed :("
    failed_runs = 2

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        config=None,
        split=None,
        content={},
        dataset_git_revision=revision,
        job_runner_version=processing_graph.get_processing_step("dataset-config-names").job_runner_version,
        progress=1.0,
        http_status=HTTPStatus.OK,
        failed_runs=failed_runs,
    )

    queue = Queue()
    assert JobDocument.objects().count() == 0
    queue.add_job(
        job_type="dataset-config-names",
        dataset=dataset,
        revision=revision,
        priority=Priority.NORMAL,
        difficulty=50,
    )
    job_info = queue.start_job()

    job_runner = DummyJobRunner(
        job_info=job_info,
        app_config=app_config,
    )

    job_manager = JobManager(job_info=job_info, app_config=app_config, job_runner=job_runner)

    job_manager.set_crashed(message=message)
    response = CachedResponseDocument.objects()[0]
    expected_error = {"error": message}
    assert response.http_status == HTTPStatus.NOT_IMPLEMENTED
    assert response.error_code == "JobManagerCrashedError"
    assert response.dataset == dataset
    assert response.dataset_git_revision == revision
    assert response.content == expected_error
    assert response.details == expected_error
    assert response.failed_runs == failed_runs + 1
    # TODO: check if it stores the correct dataset git sha and job version when it's implemented

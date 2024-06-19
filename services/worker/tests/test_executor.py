import os
import sys
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
import pytz
from filelock import FileLock
from libcommon.dtos import JobInfo, Priority, Status
from libcommon.queue import JobDocument, JobDoesNotExistError, Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedResponseDocument
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from libcommon.utils import get_datetime
from mirakuru import ProcessExitedWithError, TimeoutExpired
from pytest import fixture

from worker.config import AppConfig
from worker.executor import WorkerExecutor
from worker.job_runner_factory import JobRunnerFactory
from worker.loop import WorkerState
from worker.resources import LibrariesResource

_TIME = int(os.environ.get("WORKER_TEST_TIME", int(time.time() * 10e3)))


def get_job_info(prefix: str = "base") -> JobInfo:
    job_id = prefix.encode().hex()
    assert len(job_id) <= 24, "please choose a smaller prefix"
    return JobInfo(
        job_id=job_id + "0" * (24 - len(job_id)),
        type="dataset-config-names",
        params={
            "dataset": f"DVUser/{prefix}_dataset_{_TIME}",
            "revision": "revision",
            "config": "default",
            "split": "train",
        },
        priority=Priority.LOW,
        difficulty=50,
        started_at=None,
    )


def write_worker_state(worker_state: WorkerState, worker_state_file_path: str) -> None:
    with FileLock(worker_state_file_path + ".lock"):
        with open(worker_state_file_path, "wb") as worker_state_f:
            worker_state_f.write(orjson.dumps(worker_state))


def start_worker_loop() -> None:
    app_config = AppConfig.from_env()
    if not app_config.worker.state_file_path:
        raise ValueError("Failed to get worker state because 'state_file_path' is missing.")
    if "--print-worker-state-path" in sys.argv:
        print(app_config.worker.state_file_path, flush=True)
    current_job_info = get_job_info()
    worker_state = WorkerState(current_job_info=current_job_info, last_updated=get_datetime())
    write_worker_state(worker_state, app_config.worker.state_file_path)


def start_worker_loop_that_crashes() -> None:
    app_config = AppConfig.from_env()
    if not app_config.worker.state_file_path:
        raise ValueError("Failed to get worker state because 'state_file_path' is missing.")
    if "--print-worker-state-path" in sys.argv:
        print(app_config.worker.state_file_path, flush=True)
    raise RuntimeError("Tried to run a bad worker loop")


def start_worker_loop_that_times_out() -> None:
    time.sleep(20)


def start_worker_loop_with_long_job() -> None:
    app_config = AppConfig.from_env()
    if not app_config.worker.state_file_path:
        raise ValueError("Failed to get worker state because 'state_file_path' is missing.")
    if "--print-worker-state-path" in sys.argv:
        print(app_config.worker.state_file_path, flush=True)
    current_job_info = get_job_info("long")
    with QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url):
        current_job = JobDocument.objects(pk=current_job_info["job_id"]).get()
        assert current_job.started_at is not None
        worker_state = WorkerState(
            current_job_info=current_job_info, last_updated=pytz.UTC.localize(current_job.started_at)
        )
        if current_job.status == Status.STARTED:
            write_worker_state(worker_state, app_config.worker.state_file_path)
            time.sleep(20)
            Queue().finish_job(current_job_info["job_id"])


@fixture
def set_worker_state(worker_state_file_path: str) -> Iterator[WorkerState]:
    job_info = get_job_info()
    worker_state = WorkerState(current_job_info=job_info, last_updated=get_datetime())
    write_worker_state(worker_state, worker_state_file_path)
    yield worker_state
    os.remove(worker_state_file_path)


@fixture
def set_just_started_job_in_queue(queue_mongo_resource: QueueMongoResource) -> Iterator[JobDocument]:
    job_info = get_job_info()
    try:
        JobDocument.get(job_id=job_info["job_id"]).delete()
    except JobDoesNotExistError:
        pass
    created_at = get_datetime()
    job = JobDocument(
        pk=job_info["job_id"],
        type=job_info["type"],
        dataset=job_info["params"]["dataset"],
        revision=job_info["params"]["revision"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
        unicity_id="unicity_id",
        namespace="user",
        priority=job_info["priority"],
        status=Status.STARTED,
        created_at=created_at,
        started_at=created_at + timedelta(microseconds=1),
        difficulty=job_info["difficulty"],
    )
    job.save()
    yield job
    job.delete()


@fixture
def set_long_running_job_in_queue(
    app_config: AppConfig, queue_mongo_resource: QueueMongoResource
) -> Iterator[JobDocument]:
    job_info = get_job_info("long")
    try:
        JobDocument.get(job_id=job_info["job_id"]).delete()
    except JobDoesNotExistError:
        pass
    created_at = get_datetime() - timedelta(days=1)
    last_heartbeat = get_datetime() - timedelta(seconds=app_config.worker.heartbeat_interval_seconds)
    job = JobDocument(
        pk=job_info["job_id"],
        type=job_info["type"],
        dataset=job_info["params"]["dataset"],
        revision=job_info["params"]["revision"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
        unicity_id="unicity_id",
        namespace="user",
        priority=job_info["priority"],
        status=Status.STARTED,
        created_at=created_at,
        started_at=created_at + timedelta(milliseconds=1),
        last_heartbeat=last_heartbeat,
        difficulty=job_info["difficulty"],
    )
    job.save()
    yield job
    job.delete()


@fixture
def set_zombie_job_in_queue(queue_mongo_resource: QueueMongoResource) -> Iterator[JobDocument]:
    if not queue_mongo_resource.is_available():
        raise RuntimeError("Mongo resource is not available")
    job_info = get_job_info("zombie")
    try:
        JobDocument.get(job_id=job_info["job_id"]).delete()
    except JobDoesNotExistError:
        pass
    created_at = get_datetime() - timedelta(days=1)
    job = JobDocument(
        pk=job_info["job_id"],
        type=job_info["type"],
        dataset=job_info["params"]["dataset"],
        revision=job_info["params"]["revision"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
        unicity_id="unicity_id",
        namespace="user",
        priority=job_info["priority"],
        status=Status.STARTED,
        created_at=created_at,
        started_at=created_at + timedelta(milliseconds=1),
        last_heartbeat=created_at + timedelta(milliseconds=2),
        difficulty=job_info["difficulty"],
    )
    job.save()
    yield job
    job.delete()


@fixture
def job_runner_factory(
    app_config: AppConfig,
    libraries_resource: LibrariesResource,
    parquet_metadata_directory: StrPath,
    duckdb_index_cache_directory: StrPath,
    statistics_cache_directory: StrPath,
    tmp_path: Path,
) -> JobRunnerFactory:
    storage_client = StorageClient(
        protocol="file",
        storage_root=str(tmp_path / "assets"),
        base_url=app_config.assets.base_url,
        overwrite=True,  # all the job runners will overwrite the files
    )
    return JobRunnerFactory(
        app_config=app_config,
        hf_datasets_cache=libraries_resource.hf_datasets_cache,
        parquet_metadata_directory=parquet_metadata_directory,
        duckdb_index_cache_directory=duckdb_index_cache_directory,
        statistics_cache_directory=statistics_cache_directory,
        storage_client=storage_client,
    )


@fixture
def executor(
    app_config: AppConfig,
    job_runner_factory: JobRunnerFactory,
    worker_state_file_path: str,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> Iterator[WorkerExecutor]:
    worker_executor = WorkerExecutor(app_config, job_runner_factory, state_file_path=worker_state_file_path)
    yield worker_executor
    worker_executor.stop()


def test_executor_get_state(
    executor: WorkerExecutor,
    set_worker_state: WorkerState,
) -> None:
    assert executor.get_state() == set_worker_state


def test_executor_get_empty_state(
    executor: WorkerExecutor,
) -> None:
    assert executor.get_state() is None


def test_executor_heartbeat(
    executor: WorkerExecutor,
    set_just_started_job_in_queue: JobDocument,
    set_worker_state: WorkerState,
) -> None:
    current_job = set_just_started_job_in_queue
    assert current_job.last_heartbeat is None
    executor.heartbeat()
    current_job.reload()
    assert current_job.last_heartbeat is not None
    last_heartbeat_datetime = pytz.UTC.localize(current_job.last_heartbeat)
    assert last_heartbeat_datetime >= get_datetime() - timedelta(seconds=1)


def test_executor_kill_zombies(
    executor: WorkerExecutor,
    set_just_started_job_in_queue: JobDocument,
    set_long_running_job_in_queue: JobDocument,
    set_zombie_job_in_queue: JobDocument,
    tmp_dataset_repo_factory: Callable[[str], str],
) -> None:
    zombie = set_zombie_job_in_queue
    normal_job = set_just_started_job_in_queue
    tmp_dataset_repo_factory(zombie.dataset)
    executor.kill_zombies()
    assert JobDocument.objects(pk=zombie.pk).count() == 0
    assert JobDocument.objects(pk=normal_job.pk).get().status == Status.STARTED
    response = CachedResponseDocument.objects()[0]
    expected_error = {
        "error": "Job manager crashed while running this job (missing heartbeats).",
    }
    assert response.http_status == HTTPStatus.NOT_IMPLEMENTED
    assert response.error_code == "JobManagerCrashedError"
    assert response.dataset == zombie.dataset
    assert response.config == zombie.config
    assert response.split == zombie.split
    assert response.content == expected_error
    assert response.details == expected_error


@pytest.mark.asyncio
def test_executor_start(
    executor: WorkerExecutor,
    set_just_started_job_in_queue: JobDocument,
    set_zombie_job_in_queue: JobDocument,
    tmp_dataset_repo_factory: Callable[[str], str],
) -> None:
    zombie = set_zombie_job_in_queue
    tmp_dataset_repo_factory(zombie.dataset)
    # tmp_dataset_repo_factory(zombie.dataset)
    with patch.object(executor, "heartbeat", wraps=executor.heartbeat) as heartbeat_mock:
        with patch.object(executor, "kill_zombies", wraps=executor.kill_zombies) as kill_zombies_mock:
            with (
                patch("worker.executor.START_WORKER_LOOP_PATH", __file__),
                patch.dict(os.environ, {"WORKER_TEST_TIME": str(_TIME)}),
            ):
                executor.start()
    current_job = set_just_started_job_in_queue
    assert current_job is not None
    assert str(current_job.pk) == get_job_info()["job_id"]
    assert heartbeat_mock.call_count > 0
    assert JobDocument.objects(pk=set_just_started_job_in_queue.pk).get().last_heartbeat is not None
    assert kill_zombies_mock.call_count > 0
    assert JobDocument.objects(pk=set_zombie_job_in_queue.pk).count() == 0


@pytest.mark.parametrize(
    "bad_worker_loop_type", ["start_worker_loop_that_crashes", "start_worker_loop_that_times_out"]
)
def test_executor_raises_on_bad_worker(executor: WorkerExecutor, bad_worker_loop_type: str) -> None:
    with patch.dict(os.environ, {"WORKER_LOOP_TYPE": bad_worker_loop_type}):
        with (
            patch("worker.executor.START_WORKER_LOOP_PATH", __file__),
            patch.dict(os.environ, {"WORKER_TEST_TIME": str(_TIME)}),
        ):
            with pytest.raises((ProcessExitedWithError, TimeoutExpired)):
                executor.start()


def test_executor_stops_on_long_job(
    executor: WorkerExecutor,
    tmp_dataset_repo_factory: Callable[[str], str],
    set_long_running_job_in_queue: JobDocument,
    set_just_started_job_in_queue: JobDocument,
) -> None:
    long_job = set_long_running_job_in_queue
    tmp_dataset_repo_factory(long_job.dataset)
    with patch.dict(os.environ, {"WORKER_LOOP_TYPE": "start_worker_loop_with_long_job"}):
        with patch.object(executor, "max_seconds_without_heartbeat_for_zombies", -1):  # don't kill normal_job
            with patch.object(
                executor, "kill_long_job_interval_seconds", 0.1
            ):  # make sure it has the time to kill the job
                with (
                    patch("worker.executor.START_WORKER_LOOP_PATH", __file__),
                    patch.dict(os.environ, {"WORKER_TEST_TIME": str(_TIME)}),
                ):
                    executor.start()

    assert long_job is not None
    assert str(long_job.pk) == get_job_info("long")["job_id"]
    assert JobDocument.objects(pk=long_job.pk).count() == 0, "must be deleted because too long"

    responses = CachedResponseDocument.objects(error_code="JobManagerExceededMaximumDurationError")
    assert len(responses) == 1
    response = responses[0]
    expected_error = {
        "error": "Job manager was killed while running this job (job exceeded maximum duration).",
    }
    assert response.http_status == HTTPStatus.NOT_IMPLEMENTED
    assert response.error_code == "JobManagerExceededMaximumDurationError"
    assert response.dataset == long_job.dataset
    assert response.config == long_job.config
    assert response.split == long_job.split
    assert response.content == expected_error
    assert response.details == expected_error


if __name__ == "__main__":
    worker_loop_type = os.environ.get("WORKER_LOOP_TYPE", "start_worker_loop")
    if worker_loop_type == "start_worker_loop_that_crashes":
        start_worker_loop_that_crashes()
    elif worker_loop_type == "start_worker_loop_that_times_out":
        start_worker_loop_that_times_out()
    elif worker_loop_type == "start_worker_loop_with_long_job":
        start_worker_loop_with_long_job()
    else:
        start_worker_loop()

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from http import HTTPStatus
from typing import Optional, Union

import pandas as pd

from libcommon.constants import (
    CONFIG_INFO_KIND,
    CONFIG_SPLIT_NAMES_KIND,
    DATASET_CONFIG_NAMES_KIND,
    DEFAULT_DIFFICULTY_MAX,
    DIFFICULTY_BONUS_BY_FAILED_RUNS,
)
from libcommon.dtos import JobInfo, JobResult, Priority
from libcommon.processing_graph import ProcessingGraph, ProcessingStep, ProcessingStepDoesNotExist, processing_graph
from libcommon.prometheus import StepProfiler
from libcommon.queue import Queue
from libcommon.simple_cache import (
    CachedArtifactNotFoundError,
    delete_dataset_responses,
    fetch_names,
    get_cache_entries_df,
    get_response,
    get_response_metadata,
    upsert_response_params,
)
from libcommon.state import ArtifactState, DatasetState, FirstStepsDatasetState
from libcommon.storage_client import StorageClient

# TODO: clean dangling cache entries


@dataclass
class CacheStatus:
    cache_has_different_git_revision: dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_outdated_by_parent: dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_empty: dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_error_to_retry: dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_job_runner_obsolete: dict[str, ArtifactState] = field(default_factory=dict)
    up_to_date: dict[str, ArtifactState] = field(default_factory=dict)

    def as_response(self) -> dict[str, list[str]]:
        return {
            "cache_has_different_git_revision": sorted(self.cache_has_different_git_revision.keys()),
            "cache_is_outdated_by_parent": sorted(self.cache_is_outdated_by_parent.keys()),
            "cache_is_empty": sorted(self.cache_is_empty.keys()),
            "cache_is_error_to_retry": sorted(self.cache_is_error_to_retry.keys()),
            "cache_is_job_runner_obsolete": sorted(self.cache_is_job_runner_obsolete.keys()),
            "up_to_date": sorted(self.up_to_date.keys()),
        }


@dataclass
class QueueStatus:
    in_process: set[str] = field(default_factory=set)

    def as_response(self) -> dict[str, list[str]]:
        return {"in_process": sorted(self.in_process)}


@dataclass
class TasksStatistics:
    num_created_jobs: int = 0
    num_deleted_waiting_jobs: int = 0
    num_deleted_cache_entries: int = 0
    num_deleted_storage_directories: int = 0

    def add(self, other: "TasksStatistics") -> None:
        self.num_created_jobs += other.num_created_jobs
        self.num_deleted_waiting_jobs += other.num_deleted_waiting_jobs
        self.num_deleted_cache_entries += other.num_deleted_cache_entries
        self.num_deleted_storage_directories += other.num_deleted_storage_directories

    def has_tasks(self) -> bool:
        return any(
            [
                self.num_created_jobs > 0,
                self.num_deleted_waiting_jobs > 0,
                self.num_deleted_cache_entries > 0,
                self.num_deleted_storage_directories > 0,
            ]
        )

    def get_log(self) -> str:
        return (
            f"{self.num_created_jobs} created jobs, {self.num_deleted_waiting_jobs} deleted waiting jobs,"
            f" {self.num_deleted_cache_entries} deleted cache entries, {self.num_deleted_storage_directories} deleted"
            f" storage directories"
        )


@dataclass
class Task(ABC):
    id: str = field(init=False)
    long_id: str = field(init=False)

    @abstractmethod
    def run(self) -> TasksStatistics:
        pass


@dataclass
class CreateJobsTask(Task):
    job_infos: list[JobInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"CreateJobs,{len(self.job_infos)}"
        types = [job_info["type"] for job_info in self.job_infos]
        self.long_id = f"CreateJobs,{types}"

    def run(self) -> TasksStatistics:
        """
        Create the jobs.

        Returns:
            `TasksStatistics`: The statistics of the jobs creation.
        """
        with StepProfiler(
            method="CreateJobsTask.run",
            step="all",
        ):
            num_created_jobs = Queue().create_jobs(job_infos=self.job_infos)
            if num_created_jobs != len(self.job_infos):
                raise ValueError(
                    f"Something went wrong when creating jobs: {len(self.job_infos)} jobs were supposed to be"
                    f" created, but {num_created_jobs} were created."
                )
            return TasksStatistics(num_created_jobs=num_created_jobs)


@dataclass
class DeleteWaitingJobsTask(Task):
    jobs_df: pd.DataFrame

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"DeleteWaitingJobs,{len(self.jobs_df)}"
        types = [row["type"] for _, row in self.jobs_df.iterrows()]
        self.long_id = f"DeleteWaitingJobs,{types}"

    def run(self) -> TasksStatistics:
        """
        Delete the waiting jobs.

        Returns:
            `TasksStatistics`: The statistics of the waiting jobs deletion.
        """
        with StepProfiler(
            method="DeleteWaitingJobsTask.run",
            step="all",
        ):
            num_deleted_waiting_jobs = Queue().delete_waiting_jobs_by_job_id(job_ids=self.jobs_df["job_id"].tolist())
            logging.debug(f"{num_deleted_waiting_jobs} waiting jobs were deleted.")
            return TasksStatistics(num_deleted_waiting_jobs=num_deleted_waiting_jobs)


@dataclass
class DeleteDatasetWaitingJobsTask(Task):
    dataset: str

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"DeleteDatasetJobs,{self.dataset}"
        self.long_id = self.id

    def run(self) -> TasksStatistics:
        """
        Delete the dataset waiting jobs.

        Returns:
            `TasksStatistics`: The statistics of the waiting jobs deletion.
        """
        with StepProfiler(
            method="DeleteDatasetWaitingJobsTask.run",
            step="all",
        ):
            return TasksStatistics(num_deleted_waiting_jobs=Queue().delete_dataset_waiting_jobs(dataset=self.dataset))


@dataclass
class DeleteDatasetCacheEntriesTask(Task):
    dataset: str

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"DeleteDatasetCacheEntries,{self.dataset}"
        self.long_id = self.id

    def run(self) -> TasksStatistics:
        """
        Delete the dataset cache entries.

        Returns:
            `TasksStatistics`: The statistics of the cache entries deletion.
        """
        with StepProfiler(
            method="DeleteDatasetCacheEntriesTask.run",
            step="all",
        ):
            return TasksStatistics(num_deleted_cache_entries=delete_dataset_responses(dataset=self.dataset))


@dataclass
class DeleteDatasetStorageTask(Task):
    dataset: str
    storage_client: StorageClient

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"DeleteDatasetStorageTask,{self.dataset},{self.storage_client}"
        self.long_id = self.id

    def run(self) -> TasksStatistics:
        """
        Delete the dataset directory from the storage.

        Returns:
            `TasksStatistics`: The statistics of the storage directory deletion.
        """
        with StepProfiler(
            method="DeleteDatasetStorageTask.run",
            step="all",
        ):
            return TasksStatistics(
                num_deleted_storage_directories=self.storage_client.delete_dataset_directory(self.dataset)
            )


SupportedTask = Union[
    CreateJobsTask,
    DeleteWaitingJobsTask,
    DeleteDatasetWaitingJobsTask,
    DeleteDatasetCacheEntriesTask,
    DeleteDatasetStorageTask,
]


@dataclass
class Plan:
    tasks: list[SupportedTask] = field(init=False)

    def __post_init__(self) -> None:
        self.tasks = []

    def add_task(self, task: SupportedTask) -> None:
        self.tasks.append(task)

    def run(self) -> TasksStatistics:
        """Run all the tasks in the plan.

        Returns:
            `TasksStatistics`: The statistics of the plan (sum of the statistics of the tasks).
        """
        statistics = TasksStatistics()
        for idx, task in enumerate(self.tasks):
            logging.debug(f"Running task [{idx}/{len(self.tasks)}]: {task.long_id}")
            statistics.add(task.run())
        return statistics

    def as_response(self) -> list[str]:
        return sorted(task.id for task in self.tasks)


def get_num_bytes_from_config_infos(dataset: str, config: str, split: Optional[str] = None) -> Optional[int]:
    try:
        resp = get_response(kind=CONFIG_INFO_KIND, dataset=dataset, config=config)
    except CachedArtifactNotFoundError:
        return None
    if "dataset_info" in resp["content"] and isinstance(resp["content"]["dataset_info"], dict):
        dataset_info = resp["content"]["dataset_info"]
        if split is None:
            num_bytes = dataset_info.get("dataset_size")
            if isinstance(num_bytes, int):
                return num_bytes
        elif "splits" in dataset_info and isinstance(dataset_info["splits"], dict):
            split_infos = dataset_info["splits"]
            if split in split_infos and isinstance(split_infos[split], dict):
                split_info = split_infos[split]
                num_bytes = split_info.get("num_bytes")
                if isinstance(num_bytes, int):
                    return num_bytes
    return None


@dataclass
class AfterJobPlan(Plan):
    """
    Plan to create jobs after a processing step has finished.

    Args:
        job_info (`JobInfo`): The job info.
        processing_graph (`ProcessingGraph`): The processing graph.
    """

    job_info: JobInfo
    processing_graph: ProcessingGraph
    failed_runs: int

    dataset: str = field(init=False)
    config: Optional[str] = field(init=False)
    split: Optional[str] = field(init=False)
    revision: str = field(init=False)
    priority: Priority = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dataset = self.job_info["params"]["dataset"]
        self.revision = self.job_info["params"]["revision"]
        self.priority = self.job_info["priority"]

        config = self.job_info["params"]["config"]
        split = self.job_info["params"]["split"]
        job_type = self.job_info["type"]
        try:
            processing_step = self.processing_graph.get_processing_step_by_job_type(job_type)
            next_processing_steps = self.processing_graph.get_children(processing_step.name)
        except ProcessingStepDoesNotExist as e:
            raise ValueError(f"Processing step with job type: {job_type} does not exist") from e

        if len(next_processing_steps) == 0:
            # no next processing step, nothing to do
            return

        # get the dataset infos to estimate difficulty
        if config is not None:
            self.num_bytes = get_num_bytes_from_config_infos(dataset=self.dataset, config=config, split=split)
        else:
            self.num_bytes = None

        # get the list of pending jobs for the children
        # note that it can contain a lot of unrelated jobs, we will clean after
        self.pending_jobs_df = Queue().get_pending_jobs_df(
            dataset=self.dataset,
            job_types=[next_processing_step.job_type for next_processing_step in next_processing_steps],
        )

        self.job_infos_to_create: list[JobInfo] = []
        config_names: Optional[list[str]] = None
        split_names: Optional[list[str]] = None

        # filter to only get the jobs that are not already in the queue
        for next_processing_step in next_processing_steps:
            if processing_step.input_type == next_processing_step.input_type:
                # same level, one job is expected
                # D -> D, C -> C, S -> S
                self.update(next_processing_step, config, split)
            elif processing_step.input_type in ["config", "split"] and next_processing_step.input_type == "dataset":
                # going to upper level (fan-in), one job is expected
                # S -> D, C -> D
                self.update(next_processing_step, None, None)
            elif processing_step.input_type == "split" and next_processing_step.input_type == "config":
                # going to upper level (fan-in), one job is expected
                # S -> C
                self.update(next_processing_step, config, None)
            elif processing_step.input_type == "dataset" and next_processing_step.input_type == "config":
                # going to lower level (fan-out), one job is expected per config, we need the list of configs
                # D -> C
                if config_names is None:
                    config_names = fetch_names(
                        dataset=self.dataset,
                        config=None,
                        cache_kind=DATASET_CONFIG_NAMES_KIND,
                        names_field="config_names",
                        name_field="config",
                    )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)
                for config_name in config_names:
                    self.update(next_processing_step, config_name, None)
            elif processing_step.input_type == "config" and next_processing_step.input_type == "split":
                # going to lower level (fan-out), one job is expected per split, we need the list of splits
                # C -> S
                if split_names is None:
                    split_names = fetch_names(
                        dataset=self.dataset,
                        config=config,
                        cache_kind=CONFIG_SPLIT_NAMES_KIND,
                        names_field="splits",
                        name_field="split",
                    )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)
                for split_name in split_names:
                    self.update(next_processing_step, config, split_name)
            else:
                raise NotImplementedError(
                    f"Unsupported input types: {processing_step.input_type} -> {next_processing_step.input_type}"
                )
                # we don't support fan-out dataset-level to split-level (no need for now)

        # Better keep this order: delete, then create
        # Note that all the waiting jobs for other revisions will be deleted
        # The started jobs are ignored, for now.
        if not self.pending_jobs_df.empty:
            self.add_task(DeleteWaitingJobsTask(jobs_df=self.pending_jobs_df))
        if self.job_infos_to_create:
            self.add_task(CreateJobsTask(job_infos=self.job_infos_to_create))

    def update(
        self,
        next_processing_step: ProcessingStep,
        config: Optional[str],
        split: Optional[str],
    ) -> None:
        # ignore unrelated jobs
        config_mask = (
            self.pending_jobs_df["config"].isnull() if config is None else self.pending_jobs_df["config"] == config
        )
        split_mask = (
            self.pending_jobs_df["split"].isnull() if split is None else self.pending_jobs_df["split"] == split
        )

        unrelated_jobs_mask = (self.pending_jobs_df["type"] == next_processing_step.job_type) & (
            (self.pending_jobs_df["dataset"] != self.dataset) | (~config_mask) | (~split_mask)
        )
        self.pending_jobs_df = self.pending_jobs_df[~unrelated_jobs_mask]

        jobs_mask = (
            (self.pending_jobs_df["type"] == next_processing_step.job_type)
            & (self.pending_jobs_df["dataset"] == self.dataset)
            & (config_mask)
            & (split_mask)
        )
        ok_jobs_mask = jobs_mask & (self.pending_jobs_df["revision"] == self.revision)
        if ok_jobs_mask.any():
            # remove the first ok job for the list, and keep the others to delete them later
            self.pending_jobs_df.drop(ok_jobs_mask.idxmax(), inplace=True)
        else:
            # no pending job for the current processing step
            difficulty = next_processing_step.difficulty
            if self.num_bytes is not None and self.num_bytes >= self.processing_graph.min_bytes_for_bonus_difficulty:
                difficulty += next_processing_step.bonus_difficulty_if_dataset_is_big
            # increase difficulty according to number of failed runs
            difficulty = min(DEFAULT_DIFFICULTY_MAX, difficulty)
            self.job_infos_to_create.append(
                {
                    "job_id": "not used",  # TODO: remove this field
                    "type": next_processing_step.job_type,
                    "params": {
                        "dataset": self.dataset,
                        "config": config,
                        "split": split,
                        "revision": self.revision,
                    },
                    "priority": self.priority,
                    "difficulty": difficulty,
                    "started_at": None,
                }
            )


@dataclass
class DatasetBackfillPlan(Plan):
    """
    Plan to backfill a dataset for a given revision.

    The plan is composed of tasks to delete and create jobs.

    Args:
        dataset: dataset name
        revision: revision to backfill
        priority: priority of the jobs to create
        only_first_processing_steps: if True, only the first processing steps are backfilled
        processing_graph: processing graph
    """

    dataset: str
    revision: str
    priority: Priority = Priority.LOW
    only_first_processing_steps: bool = False
    processing_graph: ProcessingGraph = field(default=processing_graph)

    pending_jobs_df: pd.DataFrame = field(init=False)
    cache_entries_df: pd.DataFrame = field(init=False)
    dataset_state: DatasetState = field(init=False)
    cache_status: CacheStatus = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        with StepProfiler(
            method="DatasetBackfillPlan.__post_init__",
            step="all",
        ):
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="get_pending_jobs_df",
            ):
                job_types = (
                    [
                        processing_step.job_type
                        for processing_step in self.processing_graph.get_first_processing_steps()
                    ]
                    if self.only_first_processing_steps
                    else None
                )
                self.pending_jobs_df = Queue().get_pending_jobs_df(
                    dataset=self.dataset,
                    job_types=job_types,
                )
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="get_cache_entries_df",
            ):
                cache_kinds = (
                    [
                        processing_step.cache_kind
                        for processing_step in self.processing_graph.get_first_processing_steps()
                    ]
                    if self.only_first_processing_steps
                    else None
                )
                self.cache_entries_df = get_cache_entries_df(
                    dataset=self.dataset,
                    cache_kinds=cache_kinds,
                )

            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="get_dataset_state",
            ):
                self.dataset_state = (
                    FirstStepsDatasetState(
                        dataset=self.dataset,
                        processing_graph=self.processing_graph,
                        revision=self.revision,
                        pending_jobs_df=self.pending_jobs_df,
                        cache_entries_df=self.cache_entries_df,
                    )
                    if self.only_first_processing_steps
                    else DatasetState(
                        dataset=self.dataset,
                        processing_graph=self.processing_graph,
                        revision=self.revision,
                        pending_jobs_df=self.pending_jobs_df,
                        cache_entries_df=self.cache_entries_df,
                    )
                )
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="_get_cache_status",
            ):
                self.cache_status = self._get_cache_status()
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="_create_plan",
            ):
                self._create_plan()

    def _get_artifact_states_for_step(
        self, processing_step: ProcessingStep, config: Optional[str] = None, split: Optional[str] = None
    ) -> list[ArtifactState]:
        """Get the artifact states for a step.

        Args:
            processing_step (`ProcessingStep`): the processing step
            config (`str`, *optional*): if not None, and step input type is config or split, only return the artifact
              states for this config
            split (`str`, *optional*): if not None, and step input type is split, only return the artifact states for
              this split (config must be specified)

        Returns:
            `list[ArtifactState]`: the artifact states for the step
        """
        if processing_step.input_type == "dataset":
            artifact_states = [self.dataset_state.artifact_state_by_step[processing_step.name]]
        elif processing_step.input_type == "config":
            if config is None:
                artifact_states = [
                    config_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                ]
            else:
                artifact_states = [
                    config_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    if config_state.config == config
                ]
        elif processing_step.input_type == "split":
            if config is None:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    for split_state in config_state.split_states
                ]
            elif split is None:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    if config_state.config == config
                    for split_state in config_state.split_states
                ]
            else:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    if config_state.config == config
                    for split_state in config_state.split_states
                    if split_state.split == split
                ]
        else:
            raise ValueError(f"Invalid input type: {processing_step.input_type}")
        artifact_states_ids = {artifact_state.id for artifact_state in artifact_states}
        if len(artifact_states_ids) != len(artifact_states):
            raise ValueError(f"Duplicate artifact states for processing_step {processing_step}")
        return artifact_states

    def _get_cache_status(self) -> CacheStatus:
        cache_status = CacheStatus()

        processing_steps = (
            self.processing_graph.get_first_processing_steps()
            if self.only_first_processing_steps
            else self.processing_graph.get_topologically_ordered_processing_steps()
        )
        for processing_step in processing_steps:
            # Every step can have one or multiple artifacts, for example config-level steps have one artifact per
            # config
            artifact_states = self._get_artifact_states_for_step(processing_step)
            for artifact_state in artifact_states:
                # any of the parents is more recent?
                if any(
                    artifact_state.cache_state.is_older_than(parent_artifact_state.cache_state)
                    for parent_step in self.processing_graph.get_parents(processing_step.name)
                    for parent_artifact_state in self._get_artifact_states_for_step(
                        processing_step=parent_step,
                        config=artifact_state.config,
                        split=artifact_state.split,
                    )
                ):
                    cache_status.cache_is_outdated_by_parent[artifact_state.id] = artifact_state
                    continue

                # is empty?
                if artifact_state.cache_state.is_empty():
                    cache_status.cache_is_empty[artifact_state.id] = artifact_state
                    continue

                # is an error that can be retried?
                if artifact_state.cache_state.is_error_to_retry():
                    cache_status.cache_is_error_to_retry[artifact_state.id] = artifact_state
                    continue

                # was created with an obsolete version of the job runner?
                if artifact_state.cache_state.is_job_runner_obsolete():
                    cache_status.cache_is_job_runner_obsolete[artifact_state.id] = artifact_state
                    continue

                # has a different git revision from the dataset current revision?
                if artifact_state.cache_state.is_git_revision_different_from(self.revision):
                    cache_status.cache_has_different_git_revision[artifact_state.id] = artifact_state
                    continue

                # ok
                cache_status.up_to_date[artifact_state.id] = artifact_state

        return cache_status

    def get_queue_status(self) -> QueueStatus:
        processing_steps = (
            self.processing_graph.get_first_processing_steps()
            if self.only_first_processing_steps
            else self.processing_graph.get_topologically_ordered_processing_steps()
        )
        return QueueStatus(
            in_process={
                artifact_state.id
                for processing_step in processing_steps
                for artifact_state in self._get_artifact_states_for_step(processing_step)
                if artifact_state.job_state.is_in_process
            }
        )

    def _create_plan(self) -> None:
        pending_jobs_to_delete_df = self.pending_jobs_df.copy()
        job_infos_to_create: list[JobInfo] = []
        artifact_states = (
            list(self.cache_status.cache_is_empty.values())
            + list(self.cache_status.cache_is_error_to_retry.values())
            + list(self.cache_status.cache_is_outdated_by_parent.values())
            + list(self.cache_status.cache_is_job_runner_obsolete.values())
            + list(self.cache_status.cache_has_different_git_revision.values())
        )

        @lru_cache
        def is_big(config: str) -> bool:
            num_bytes = get_num_bytes_from_config_infos(dataset=self.dataset, config=config)
            if num_bytes is None:
                return False
            else:
                return num_bytes > self.processing_graph.min_bytes_for_bonus_difficulty

        for artifact_state in artifact_states:
            valid_pending_jobs_df = artifact_state.job_state.valid_pending_jobs_df
            if valid_pending_jobs_df.empty:
                difficulty = artifact_state.processing_step.difficulty
                if isinstance(artifact_state.config, str) and is_big(config=artifact_state.config):
                    difficulty += artifact_state.processing_step.bonus_difficulty_if_dataset_is_big
                if artifact_state.cache_state.cache_entry_metadata is not None:
                    failed_runs = artifact_state.cache_state.cache_entry_metadata["failed_runs"]
                else:
                    failed_runs = 0
                # increase difficulty according to number of failed runs
                difficulty = min(DEFAULT_DIFFICULTY_MAX, difficulty + failed_runs * DIFFICULTY_BONUS_BY_FAILED_RUNS)
                job_infos_to_create.append(
                    {
                        "job_id": "not used",
                        "type": artifact_state.processing_step.job_type,
                        "params": {
                            "dataset": self.dataset,
                            "revision": self.revision,
                            "config": artifact_state.config,
                            "split": artifact_state.split,
                        },
                        "priority": self.priority,
                        "difficulty": difficulty,
                        "started_at": None,
                    }
                )
            else:
                pending_jobs_to_delete_df.drop(valid_pending_jobs_df.index, inplace=True)
        # Better keep this order: delete, then create
        # Note that all the waiting jobs for other revisions will be deleted
        # The started jobs are ignored, for now.
        if not pending_jobs_to_delete_df.empty:
            self.add_task(DeleteWaitingJobsTask(jobs_df=pending_jobs_to_delete_df))
        if job_infos_to_create:
            self.add_task(CreateJobsTask(job_infos=job_infos_to_create))


@dataclass
class DatasetRemovalPlan(Plan):
    """
    Plan to remove a dataset.

    The plan is composed of tasks to delete jobs and cache entries.

    Args:
        dataset: dataset name
        storage_clients (`list[StorageClient]`, *optional*): The storage clients.
    """

    dataset: str
    storage_clients: Optional[list[StorageClient]]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.add_task(DeleteDatasetWaitingJobsTask(dataset=self.dataset))
        self.add_task(DeleteDatasetCacheEntriesTask(dataset=self.dataset))
        if self.storage_clients:
            for storage_client in self.storage_clients:
                self.add_task(DeleteDatasetStorageTask(dataset=self.dataset, storage_client=storage_client))


def remove_dataset(dataset: str, storage_clients: Optional[list[StorageClient]] = None) -> TasksStatistics:
    """
    Remove the dataset from the dataset viewer

    Args:
        dataset (`str`): The name of the dataset.
        storage_clients (`list[StorageClient]`, *optional*): The storage clients.

    Returns:
        `TasksStatistics`: The statistics of the deletion.
    """
    plan = DatasetRemovalPlan(dataset=dataset, storage_clients=storage_clients)
    return plan.run()
    # assets and cached_assets are deleted by the storage clients
    # TODO: delete the other files: metadata parquet, parquet, duckdb index, etc
    # note that it's not as important as the assets, because generally, we want to delete a dataset
    # form the dataset viewer because the repository does not exist anymore on the Hub, so: the other files
    # don't exist anymore either (they were in refs/convert/parquet or refs/convert/duckdb).
    # Only exception I see is when we stop supporting a dataset (blocked, disabled viewer, private dataset
    # and the user is not pro anymore, etc.)


def set_revision(
    dataset: str,
    revision: str,
    priority: Priority,
    processing_graph: ProcessingGraph = processing_graph,
) -> TasksStatistics:
    """
    Set the current revision of the dataset.

    If the revision is already set to the same value, this is a no-op. Else: one job is created for every first
        step.

    Args:
        dataset (`str`): The name of the dataset.
        revision (`str`): The new revision of the dataset.
        priority (`Priority`): The priority of the jobs to create.
        processing_graph (`ProcessingGraph`, *optional*): The processing graph.

    Returns:
        `TasksStatistics`: The statistics of the set_revision.
    """
    logging.info(f"Analyzing {dataset}")
    plan = DatasetBackfillPlan(
        dataset=dataset,
        revision=revision,
        priority=priority,
        processing_graph=processing_graph,
        only_first_processing_steps=True,
    )
    logging.info(f"Applying set_revision plan on {dataset}: plan={plan.as_response()}")
    return plan.run()


def backfill(
    dataset: str,
    revision: str,
    priority: Priority,
    processing_graph: ProcessingGraph = processing_graph,
) -> TasksStatistics:
    """
    Analyses the dataset and backfills it with all missing bits, if requires.

    Args:
        dataset (`str`): The name of the dataset.
        revision (`str`): The new revision of the dataset.
        priority (`Priority`): The priority of the jobs to create.
        processing_graph (`ProcessingGraph`, *optional*): The processing graph.

    Returns:
        `TasksStatistics`: The statistics of the backfill.
    """
    logging.info(f"Analyzing {dataset}")
    plan = DatasetBackfillPlan(
        dataset=dataset,
        revision=revision,
        priority=priority,
        processing_graph=processing_graph,
        only_first_processing_steps=False,
    )
    logging.info(f"Applying backfill plan on {dataset}: plan={plan.as_response()}")
    return plan.run()


def finish_job(
    job_result: JobResult,
    processing_graph: ProcessingGraph = processing_graph,
) -> TasksStatistics:
    """
    Finish a job.

    It will finish the job, store the result in the cache, and trigger the next steps.

    Args:
        job_result (`JobResult`): The result of the job.
        processing_graph (`ProcessingGraph`, *optional*): The processing graph.

    Raises:
        [`ValueError`]: If the job is not found, or if the processing step is not found.

    Returns:
        `TasksStatistics`: The statistics of the finish_job.
    """
    # check if the job is still in started status
    job_info = job_result["job_info"]
    if not Queue().is_job_started(job_id=job_info["job_id"]):
        logging.debug("the job was deleted, don't update the cache")
        return TasksStatistics()
    # if the job could not provide an output, finish it and return
    if not job_result["output"]:
        Queue().finish_job(job_id=job_info["job_id"])
        logging.debug("the job raised an exception, don't update the cache")
        return TasksStatistics()
    # update the cache
    output = job_result["output"]
    params = job_info["params"]
    try:
        processing_step = processing_graph.get_processing_step_by_job_type(job_info["type"])
    except ProcessingStepDoesNotExist as e:
        raise ValueError(f"Processing step for job type {job_info['type']} does not exist") from e

    try:
        previous_response = get_response_metadata(
            kind=processing_step.cache_kind, dataset=params["dataset"], config=params["config"], split=params["split"]
        )
        failed_runs = (
            previous_response["failed_runs"] + 1
            if output["http_status"] != HTTPStatus.OK
            and previous_response["dataset_git_revision"] == params["revision"]
            else 0
        )
    except CachedArtifactNotFoundError:
        failed_runs = 0

    upsert_response_params(
        # inputs
        kind=processing_step.cache_kind,
        job_params=params,
        job_runner_version=job_result["job_runner_version"],
        # output
        content=output["content"],
        http_status=output["http_status"],
        error_code=output["error_code"],
        details=output["details"],
        progress=output["progress"],
        started_at=job_info["started_at"],
        failed_runs=failed_runs,
    )
    logging.debug("the job output has been written to the cache.")
    # finish the job
    Queue().finish_job(job_id=job_info["job_id"])
    logging.debug("the job has been finished.")
    # trigger the next steps
    plan = AfterJobPlan(job_info=job_info, processing_graph=processing_graph, failed_runs=failed_runs)
    statistics = plan.run()
    logging.debug("jobs have been created for the next steps.")
    return statistics


def has_pending_ancestor_jobs(
    dataset: str, processing_step_name: str, processing_graph: ProcessingGraph = processing_graph
) -> bool:
    """
    Check if the processing steps, or one of their ancestors, have a pending job, ie. if artifacts could exist
        in the cache in the future. This method is used when a cache entry is missing in the API,
        to return a:
        - 404 error, saying that the artifact does not exist,
        - or a 500 error, saying that the artifact could be available soon (retry).

    It is implemented by checking if a job exists for the artifacts or one of their ancestors.

    Note that, if dataset-config-names' job is pending, we cannot know if the config is valid or not, so we
        consider that the artifact could exist.

    Args:
        dataset (`str`): The name of the dataset.
        processing_step_name (`str`): The processing step name (artifact) to check.
        processing_graph (`ProcessingGraph`, *optional*): The processing graph.

    Raises:
        [`ProcessingStepDoesNotExist`]: If any of the processing step does not exist.

    Returns:
        `bool`: True if any of the artifact could exist, False otherwise.
    """
    processing_step = processing_graph.get_processing_step(processing_step_name)
    ancestors = processing_graph.get_ancestors(processing_step_name)
    job_types = [ancestor.job_type for ancestor in ancestors] + [processing_step.job_type]
    logging.debug(f"looking at ancestor jobs of {processing_step_name}: {job_types}")
    # check if a pending job exists for the artifact or one of its ancestors
    # note that we cannot know if the ancestor is really for the artifact (ie: ancestor is for config1,
    # while we look for config2,split1). Looking in this detail would be too complex, this approximation
    # is good enough.
    return Queue().has_pending_jobs(dataset=dataset, job_types=job_types)


def get_revision(dataset: str) -> Optional[str]:
    cache_kinds = [processing_step.cache_kind for processing_step in processing_graph.get_first_processing_steps()]
    cache_entries = get_cache_entries_df(
        dataset=dataset,
        cache_kinds=cache_kinds,
    ).to_dict(orient="list")
    if cache_entries.get("dataset_git_revision") and isinstance(
        revision := cache_entries["dataset_git_revision"][0], str
    ):
        return revision
    job_types = [processing_step.job_type for processing_step in processing_graph.get_first_processing_steps()]
    pending_jobs = (
        Queue()
        .get_pending_jobs_df(
            dataset=dataset,
            job_types=job_types,
        )
        .to_dict(orient="list")
    )
    if pending_jobs.get("revision") and isinstance(revision := pending_jobs["revision"][0], str):
        return revision
    return None

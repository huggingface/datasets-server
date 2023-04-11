# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Priority, Queue
from libcommon.simple_cache import (
    CacheEntryWithoutContent,
    DoesNotExist,
    get_best_response,
    get_response,
    get_response_without_content,
)


# TODO: assets, cached_assets, parquet files
# TODO: obsolete/dangling cache entries and jobs
# TODO: report, show in endpoint
# TODO: plan what to do (backfill: create job, delete cache entries, delete assets)
# TODO: add git version
# TODO: add details about jobs (priority, force, status, times)

HARD_CODED_CONFIG_NAMES_CACHE_KIND = "/config-names"
HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND = "/split-names-from-streaming"
HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND = "/split-names-from-dataset-info"


def fetch_config_names(dataset: str) -> List[str]:
    """Fetch the list of config names from the database."""
    config_names = []

    response = get_response(HARD_CODED_CONFIG_NAMES_CACHE_KIND, dataset=dataset, config=None, split=None)
    for config_name_item in response["content"]["config_names"]:
        config_name = config_name_item["config"]
        if not isinstance(config_name, str):
            raise ValueError(f"Invalid config name: {config_name}, type should be str, got: {type(config_name)}")
        config_names.append(config_name)
    return config_names


def fetch_split_names(dataset: str, config: str) -> List[str]:
    """Fetch the list of config names from the database."""
    split_names = []

    best_response = get_best_response(
        [HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND, HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND],
        dataset=dataset,
        config=config,
        split=None,
    )
    for split_name_item in best_response.response["content"]["split_names"]:
        split_name = split_name_item["split"]
        if not isinstance(split_name, str):
            raise ValueError(f"Invalid split name: {split_name}, type should be str, got: {type(split_name)}")
        split_names.append(split_name)
    return split_names


@dataclass
class BackfillTask:
    """A backfill task."""

    job_type: str
    dataset: str
    config: Optional[str]
    split: Optional[str]

    def __post_init__(self) -> None:
        self.task = f"backfill[{self.job_type},{self.dataset},{self.config},{self.split}]"

    def __str__(self) -> str:
        return self.task

    def run(self, force: bool, priority: Priority) -> None:
        Queue().upsert_job(
            job_type=self.job_type,
            dataset=self.dataset,
            config=self.config,
            split=self.split,
            force=force,
            priority=priority,
        )


@dataclass
class JobState:
    """The state of a job for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    job_type: str
    is_in_process: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_in_process = Queue().is_job_in_process(
            job_type=self.job_type, dataset=self.dataset, config=self.config, split=self.split
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_in_process": self.is_in_process,
        }


ERROR_CODES_TO_RETRY: List[str] = []


@dataclass
class CacheState:
    """The state of a cache entry for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    cache_kind: str
    cache_entry_without_content: Optional[CacheEntryWithoutContent] = field(init=False)
    exists: bool = field(init=False)
    is_success: bool = field(init=False)

    def __post_init__(self) -> None:
        self.cache_entry_without_content = None
        with contextlib.suppress(DoesNotExist):
            self.cache_entry_without_content = get_response_without_content(
                kind=self.cache_kind, dataset=self.dataset, config=self.config, split=self.split
            )
        """Whether the cache entry exists."""
        self.exists = self.cache_entry_without_content is not None
        self.is_success = (
            self.cache_entry_without_content is not None and self.cache_entry_without_content["http_status"] < 400
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exists": self.exists,
            "is_success": self.is_success,
        }

    def should_be_refreshed(self) -> bool:
        empty = self.cache_entry_without_content is None
        error_to_retry = self.cache_entry_without_content is not None and (
            self.cache_entry_without_content["http_status"] >= 400
            and self.cache_entry_without_content["error_code"] in ERROR_CODES_TO_RETRY
        )
        # TODO: old git revision
        # TODO: old job_runner_version
        return empty or error_to_retry


@dataclass
class StepState:
    """The state of a step for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    step: ProcessingStep
    job_state: JobState = field(init=False)
    cache_state: CacheState = field(init=False)

    def __post_init__(self) -> None:
        if self.step.input_type == "dataset":
            if self.config is not None or self.split is not None:
                raise ValueError("Step input type is dataset, but config or split is not None")
        elif self.step.input_type == "config":
            if self.config is None or self.split is not None:
                raise ValueError("Step input type is config, but config is None or split is not None")
        elif self.step.input_type == "split":
            if self.config is None or self.split is None:
                raise ValueError("Step input type is split, but config or split is None")
        else:
            raise ValueError(f"Invalid step input type: {self.step.input_type}")
        self.job_state = JobState(
            job_type=self.step.job_type, dataset=self.dataset, config=self.config, split=self.split
        )
        self.cache_state = CacheState(
            cache_kind=self.step.cache_kind, dataset=self.dataset, config=self.config, split=self.split
        )

    def get_backfill_tasks(self) -> List[BackfillTask]:
        tasks: List[BackfillTask] = []
        if self.cache_state.should_be_refreshed() and not self.job_state.is_in_process:
            tasks.append(
                BackfillTask(job_type=self.step.job_type, dataset=self.dataset, config=self.config, split=self.split)
            )
        return tasks

    def backfill(self) -> None:
        """Backfill the cache entry for this step."""
        for backfill_tasks in self.get_backfill_tasks():
            backfill_tasks.run(force=True, priority=Priority.LOW)

    def should_be_backfilled(self) -> bool:
        return len(self.get_backfill_tasks()) > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step.name,
            "job_state": self.job_state.as_dict(),
            "cache_state": self.cache_state.as_dict(),
            "should_be_backfilled": self.should_be_backfilled(),
        }


@dataclass
class SplitState:
    """The state of a split."""

    dataset: str
    config: str
    split: str
    processing_graph: ProcessingGraph
    step_states: List[StepState] = field(init=False)

    def __post_init__(self) -> None:
        self.step_states = [
            StepState(step=step, dataset=self.dataset, config=self.config, split=self.split)
            for step in self.processing_graph.steps.values()
            if step.input_type == "split"
        ]

    def get_backfill_tasks(self) -> List[BackfillTask]:
        tasks: List[BackfillTask] = []
        for step_state in self.step_states:
            tasks.extend(step_state.get_backfill_tasks())
        return tasks

    def backfill(self) -> None:
        """Backfill the cache entry for this split."""
        for backfill_tasks in self.get_backfill_tasks():
            backfill_tasks.run(force=True, priority=Priority.LOW)

    def should_be_backfilled(self) -> bool:
        return len(self.get_backfill_tasks()) > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "step_states": [step_state.as_dict() for step_state in self.step_states],
            "should_be_backfilled": self.should_be_backfilled(),
        }


@dataclass
class ConfigState:
    """The state of a config."""

    dataset: str
    config: str
    processing_graph: ProcessingGraph
    split_names: List[str] = field(init=False)
    split_states: Mapping[str, SplitState] = field(init=False)
    step_states: List[StepState] = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.split_names = fetch_split_names(self.dataset, self.config)
        except Exception:
            self.split_names = []
        self.split_states = {
            split_name: SplitState(self.dataset, self.config, split_name, self.processing_graph)
            for split_name in self.split_names
        }
        self.step_states = [
            StepState(step=step, dataset=self.dataset, config=self.config, split=None)
            for step in self.processing_graph.steps.values()
            if step.input_type == "config"
        ]

    def get_backfill_tasks(self) -> List[BackfillTask]:
        tasks: List[BackfillTask] = []
        for step_state in self.step_states:
            tasks.extend(step_state.get_backfill_tasks())
        for split_state in self.split_states.values():
            tasks.extend(split_state.get_backfill_tasks())
        return tasks

    def backfill(self) -> None:
        """Backfill the cache entry for this split."""
        for backfill_tasks in self.get_backfill_tasks():
            backfill_tasks.run(force=True, priority=Priority.LOW)

    def should_be_backfilled(self) -> bool:
        return len(self.get_backfill_tasks()) > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "split_states": [split_state.as_dict() for split_state in self.split_states.values()],
            "step_states": [step_state.as_dict() for step_state in self.step_states],
            "should_be_backfilled": self.should_be_backfilled(),
        }


@dataclass
class DatasetState:
    """The state of a dataset."""

    config_states: Mapping[str, ConfigState] = field(init=False)
    dataset: str
    processing_graph: ProcessingGraph
    config_names: List[str] = field(init=False)
    step_states: List[StepState] = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.config_names = fetch_config_names(self.dataset)
        except Exception:
            self.config_names = []
        self.config_states = {
            config_name: ConfigState(dataset=self.dataset, config=config_name, processing_graph=self.processing_graph)
            for config_name in self.config_names
        }
        self.step_states = [
            StepState(step=step, dataset=self.dataset, config=None, split=None)
            for step in self.processing_graph.steps.values()
            if step.input_type == "dataset"
        ]

    def get_backfill_tasks(self) -> List[BackfillTask]:
        tasks: List[BackfillTask] = []
        for step_state in self.step_states:
            tasks.extend(step_state.get_backfill_tasks())
        for config_state in self.config_states.values():
            tasks.extend(config_state.get_backfill_tasks())
        return tasks

    def backfill(self) -> None:
        """Backfill the cache entry for this split."""
        for backfill_tasks in self.get_backfill_tasks():
            backfill_tasks.run(force=True, priority=Priority.LOW)

    def should_be_backfilled(self) -> bool:
        return len(self.get_backfill_tasks()) > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "config_states": [config_state.as_dict() for config_state in self.config_states.values()],
            "step_states": [step_state.as_dict() for step_state in self.step_states],
            "should_be_backfilled": self.should_be_backfilled(),
        }

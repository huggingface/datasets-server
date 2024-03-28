# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict, Union, get_args

import networkx as nx

from libcommon.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_INPUT_TYPE,
    DEFAULT_JOB_RUNNER_VERSION,
    MIN_BYTES_FOR_BONUS_DIFFICULTY,
)
from libcommon.utils import inputs_to_string

InputType = Literal["dataset", "config", "split"]
# ^ note that for now, the "dataset" input type means: dataset + git revision


def guard_input_type(x: Any) -> InputType:
    if x == "dataset":
        return "dataset"
    elif x == "config":
        return "config"
    elif x == "split":
        return "split"
    if x in get_args(InputType):
        raise RuntimeError(f"Value {x} should be included in the literal values")
    raise ValueError(f"Invalid input type: {x}")


def guard_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    raise ValueError(f"Invalid int: {x}")


def check_one_of_parents_is_same_or_higher_level(processing_graph: ProcessingGraph) -> None:
    first_steps = processing_graph.get_first_processing_steps()
    for step in processing_graph.get_topologically_ordered_processing_steps():
        parents = processing_graph.get_parents(step.name)
        if parents:
            parent_input_types = [p.input_type for p in parents]
            if step.input_type == "dataset":
                if "dataset" not in parent_input_types:
                    raise ValueError(
                        f"Step {step.name} is a dataset-level step but none of its parents are dataset-level."
                    )
            elif step.input_type == "config":
                if "dataset" not in parent_input_types and "config" not in parent_input_types:
                    raise ValueError(
                        f"Step {step.name} is a config-level step but none of its parents are config-level or dataset-level."
                    )
            # no need to check for splits
        else:
            # probably redundant checks
            if step not in first_steps:
                raise ValueError(f"Step {step.name} has not parents, but is not a root step.")
            if step.input_type != "dataset":
                raise ValueError(f"Step {step.name} is a root step but is not a dataset-level step.")


class ProcessingStepSpecification(TypedDict, total=False):
    input_type: InputType
    triggered_by: Union[list[str], str, None]
    job_runner_version: int
    difficulty: int
    bonus_difficulty_if_dataset_is_big: int


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


class ProcessingStepDoesNotExist(Exception):
    pass


@dataclass
class ProcessingStep:
    """A dataset processing step.

    **Attributes**:
        name (`str`): The processing step name.
        input_type (`InputType`): The input type ('dataset', 'config' or 'split').
        job_runner_version (`int`): The version of the job runner to use to compute the response.

    Getters:
        cache_kind (`str`): The cache kind (ie. the key in the cache).
        job_type (`str`): The job type (ie. the job to run to compute the response).
    """

    name: str
    input_type: InputType
    job_runner_version: int
    difficulty: int
    bonus_difficulty_if_dataset_is_big: int

    cache_kind: str = field(init=False)
    job_type: str = field(init=False)

    def __post_init__(self) -> None:
        self.cache_kind = self.name
        self.job_type = self.name

    def copy(self) -> ProcessingStep:
        """Copy the processing step.

        Returns:
            `ProcessingStep`: The copy of the processing step.
        """
        return ProcessingStep(
            name=self.name,
            input_type=self.input_type,
            job_runner_version=self.job_runner_version,
            difficulty=self.difficulty,
            bonus_difficulty_if_dataset_is_big=self.bonus_difficulty_if_dataset_is_big,
        )


def get_triggered_by_as_list(triggered_by: Union[list[str], str, None]) -> list[str]:
    if triggered_by is None:
        return []
    return [triggered_by] if isinstance(triggered_by, str) else triggered_by


def copy_processing_steps_list(processing_steps: list[ProcessingStep]) -> list[ProcessingStep]:
    return [processing_step.copy() for processing_step in processing_steps]


@dataclass
class ProcessingGraph:
    """A graph of processing steps.

    The processing steps can have multiple parents, and multiple children (next processing steps, found automatically
      by traversing the graph).
    The graph can have multiple roots.

    Args:
        specification (`ProcessingGraphSpecification`): The specification of the graph.
        min_bytes_for_bonus_difficulty (`int`, *optional*, defaults to `MIN_BYTES_FOR_BONUS_DIFFICULTY`): The minimum
            number of bytes for a dataset to be considered big and get the bonus difficulty.
        check_one_of_parents_is_same_or_higher_level (`bool`, *optional*, defaults to `True`): Whether to check that
            one of the parents of a processing step is of the same input type or a higher input type.

    Raises:
        [`ValueError`]: If the graph is not a DAG.
            If a processing step provides dataset config names but its input type is not 'dataset', or if a
            processing step provides config split names but its input type is not 'config'.
            If a root processing step (ie. a processing step with no parent) is not a dataset processing step.
    """

    specification: ProcessingGraphSpecification
    min_bytes_for_bonus_difficulty: int = MIN_BYTES_FOR_BONUS_DIFFICULTY
    check_one_of_parents_is_same_or_higher_level: bool = True

    _nx_graph: nx.DiGraph = field(init=False)
    _processing_steps: Mapping[str, ProcessingStep] = field(init=False)
    _processing_step_names_by_input_type: Mapping[InputType, list[str]] = field(init=False)
    _first_processing_steps: list[ProcessingStep] = field(init=False)
    _topologically_ordered_processing_steps: list[ProcessingStep] = field(init=False)
    _alphabetically_ordered_processing_steps: list[ProcessingStep] = field(init=False)

    def __post_init__(self) -> None:
        _nx_graph = nx.DiGraph()
        _processing_steps: dict[str, ProcessingStep] = {}
        _processing_step_names_by_input_type: dict[InputType, list[str]] = {
            "dataset": [],
            "config": [],
            "split": [],
        }
        for name, specification in self.specification.items():
            # check that the step is consistent with its specification
            input_type = guard_input_type(specification.get("input_type", DEFAULT_INPUT_TYPE))
            if (
                _nx_graph.has_node(name)
                or name in _processing_steps
                or name in _processing_step_names_by_input_type[input_type]
            ):
                raise ValueError(f"Processing step {name} is defined twice.")
            _nx_graph.add_node(name)
            _processing_steps[name] = ProcessingStep(
                name=name,
                input_type=input_type,
                job_runner_version=specification.get("job_runner_version", DEFAULT_JOB_RUNNER_VERSION),
                difficulty=specification.get("difficulty", DEFAULT_DIFFICULTY),
                bonus_difficulty_if_dataset_is_big=specification.get("bonus_difficulty_if_dataset_is_big", 0),
            )
            if _processing_steps[name].bonus_difficulty_if_dataset_is_big and input_type == "dataset":
                raise ValueError(
                    f"Processing step {name} has bonus_difficulty_if_dataset_is_big but "
                    "this field is not supported for dataset-level steps."
                )
            _processing_step_names_by_input_type[input_type].append(name)
        for name, specification in self.specification.items():
            triggered_by = get_triggered_by_as_list(specification.get("triggered_by"))
            for processing_step_name in triggered_by:
                if not _nx_graph.has_node(processing_step_name):
                    raise ValueError(
                        f"Processing step {name} is triggered by {processing_step_name} but {processing_step_name} is"
                        " not defined."
                    )
                _nx_graph.add_edge(processing_step_name, name)
        if not nx.is_directed_acyclic_graph(_nx_graph):
            raise ValueError("The graph is not a directed acyclic graph.")

        self._nx_graph = _nx_graph
        self._processing_steps = _processing_steps
        self._processing_step_names_by_input_type = _processing_step_names_by_input_type
        self._first_processing_steps = [
            self._processing_steps[processing_step_name]
            for processing_step_name, degree in _nx_graph.in_degree()
            if degree == 0
        ]
        if any(processing_step.input_type != "dataset" for processing_step in self._first_processing_steps):
            raise ValueError("The first processing steps must be dataset-level. The graph state is incoherent.")
        self._topologically_ordered_processing_steps = [
            self.get_processing_step(processing_step_name) for processing_step_name in nx.topological_sort(_nx_graph)
        ]
        self._alphabetically_ordered_processing_steps = [
            self.get_processing_step(processing_step_name) for processing_step_name in sorted(_nx_graph.nodes())
        ]
        if self.check_one_of_parents_is_same_or_higher_level:
            check_one_of_parents_is_same_or_higher_level(self)

    def get_processing_step(self, processing_step_name: str) -> ProcessingStep:
        """
        Get a processing step by its name.

        The returned processing step is a copy of the original one, so that it can be modified without affecting the
        original one.

        Args:
            processing_step_name (`str`): The name of the processing step

        Returns:
            `ProcessingStep`: The processing step
        """
        try:
            return self._processing_steps[processing_step_name].copy()
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown job type: {processing_step_name}") from e

    def get_processing_step_by_job_type(self, job_type: str) -> ProcessingStep:
        """
        Get a processing step by its job type.

        The returned processing step is a copy of the original one, so that it can be modified without affecting the
        original one.

        Args:
            job_type (`str`): The job type of the processing step

        Returns:
            `ProcessingStep`: The processing step
        """
        # for now: the job_type is just an alias for the processing step name
        return self.get_processing_step(job_type)

    def get_children(self, processing_step_name: str) -> list[ProcessingStep]:
        """
        Get the list of children processing steps

        The children processing steps are the ones that will be triggered at the end of the processing step.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            processing_step_name (`str`): The name of the processing step

        Raises:
            [`ProcessingStepDoesNotExist`]: If the processing step is not in the graph

        Returns:
            `list[ProcessingStep]`: The list of children processing steps (successors)
        """
        try:
            return [
                self.get_processing_step(successor) for successor in self._nx_graph.successors(processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_parents(self, processing_step_name: str) -> list[ProcessingStep]:
        """
        Get the list of parents processing steps

        The parent processing steps are the ones that trigger the processing step.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            processing_step_name (`str`): The name of the processing step

        Raises:
            [`ProcessingStepDoesNotExist`]: If the processing step is not in the graph

        Returns:
            `list[ProcessingStep]`: The list of parent processing steps (predecessors)
        """
        try:
            return [
                self.get_processing_step(predecessor)
                for predecessor in self._nx_graph.predecessors(processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_ancestors(self, processing_step_name: str) -> list[ProcessingStep]:
        """
        Get the list of ancestors processing steps

        The ancestor processing steps are the ones that trigger the processing step, directly or not.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            processing_step_name (`str`): The name of the processing step

        Raises:
            [`ProcessingStepDoesNotExist`]: If the processing step is not in the graph

        Returns:
            `list[ProcessingStep]`: The list of ancestor processing steps
        """
        try:
            return [
                self.get_processing_step(ancestor) for ancestor in nx.ancestors(self._nx_graph, processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_first_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the first processing steps.

        The first processing steps are the ones that don't have a previous step. This means that they will be computed
        first when a dataset is updated. Their input type is always "dataset".

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            `list[ProcessingStep]`: The list of first processing steps
        """
        return copy_processing_steps_list(self._first_processing_steps)

    def get_topologically_ordered_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps, ordered topologically.

        This means that the first processing steps are the ones that don't have a previous step, and that the last
        processing steps are the ones that don't have a next step.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            `list[ProcessingStep]`: The list of processing steps
        """
        return copy_processing_steps_list(self._topologically_ordered_processing_steps)

    def get_alphabetically_ordered_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps, ordered alphabetically by the name of the processing steps.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            `list[ProcessingStep]`: The list of processing steps
        """
        return copy_processing_steps_list(self._alphabetically_ordered_processing_steps)

    def get_processing_steps(
        self, order: Optional[Literal["alphabetical", "topological"]] = None
    ) -> list[ProcessingStep]:
        """
        Get the processing steps.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            order (`Literal["alphabetical", "topological"]`, *optional*): The order in which to return the
              processing steps. If None, the order is alphabetical. Defaults to None.

        Returns:
            `list[ProcessingStep]`: The list of processing steps
        """
        if order == "topological":
            return self.get_topologically_ordered_processing_steps()
        # default
        return self.get_alphabetically_ordered_processing_steps()

    def get_input_type_processing_steps(self, input_type: InputType = "dataset") -> list[ProcessingStep]:
        """
        Get the processing steps of input type `input_type`, in an undefined order.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            input_type (`InputType`, *optional*, defaults to `dataset`): The input type.

        Returns:
            `list[ProcessingStep]`: The list of processing steps
        """
        return [
            self.get_processing_step(processing_step_name)
            for processing_step_name in self._processing_step_names_by_input_type[input_type]
        ]


@dataclass
class Artifact:
    """An artifact."""

    processing_step: ProcessingStep
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]

    id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.processing_step.input_type == "dataset":
            if self.config is not None or self.split is not None:
                raise ValueError("Step input type is dataset, but config or split is not None")
        elif self.processing_step.input_type == "config":
            if self.config is None or self.split is not None:
                raise ValueError("Step input type is config, but config is None or split is not None")
        elif self.processing_step.input_type == "split":
            if self.config is None or self.split is None:
                raise ValueError("Step input type is split, but config or split is None")
        else:
            raise ValueError(f"Invalid step input type: {self.processing_step.input_type}")
        self.id = Artifact.get_id(
            dataset=self.dataset,
            revision=self.revision,
            config=self.config,
            split=self.split,
            processing_step_name=self.processing_step.name,
        )

    @staticmethod
    def get_id(
        dataset: str,
        revision: str,
        config: Optional[str],
        split: Optional[str],
        processing_step_name: str,
    ) -> str:
        return inputs_to_string(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            prefix=processing_step_name,
        )

    @staticmethod
    def parse_id(id: str) -> tuple[str, str, Optional[str], Optional[str], str]:
        parts = id.split(",")
        prefix = parts[0]
        parts = parts[1:]
        dataset = parts[0]
        revision = parts[1]
        parts = parts[2:]
        config = None
        split = None
        if len(parts) > 0:
            config = parts[0]
            if len(parts) > 1:
                split = parts[1]
        return dataset, revision, config, split, prefix


# Note: apart from the root steps:
# - every dataset-level step should be triggered by at least one dataset-level step
# - every config-level step should be triggered by at least one dataset-level or config-level step
# to be sure the steps are processed even if a dataset has no configs, or if a config has no splits.
specification: ProcessingGraphSpecification = {
    "dataset-config-names": {
        "input_type": "dataset",
        "job_runner_version": 1,
        "difficulty": 50,
    },
    "split-first-rows": {
        "input_type": "split",
        "triggered_by": [
            "config-split-names",  # to compute with streaming when possible
            "config-parquet-metadata",  # to compute with parquet
        ],
        "job_runner_version": 4,
        "difficulty": 70,
    },
    "config-parquet-and-info": {
        "input_type": "config",
        "triggered_by": "dataset-config-names",
        "job_runner_version": 4,
        "difficulty": 70,
    },
    "config-parquet": {
        "input_type": "config",
        "triggered_by": "config-parquet-and-info",
        "job_runner_version": 6,
        "difficulty": 20,
    },
    "config-parquet-metadata": {
        "input_type": "config",
        "triggered_by": "config-parquet",
        "job_runner_version": 2,
        "difficulty": 50,
    },
    "dataset-parquet": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-parquet",
        ],
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "config-info": {
        "input_type": "config",
        "triggered_by": "config-parquet-and-info",
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "dataset-info": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-info",
        ],
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "config-split-names": {
        "input_type": "config",
        "triggered_by": [
            "dataset-config-names",  # compute with streaming
            "config-info",  # compute from parquet conversion
        ],
        "job_runner_version": 3,
        "difficulty": 60,
    },
    "config-size": {
        "input_type": "config",
        "triggered_by": "config-parquet-and-info",
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "dataset-size": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-size",
        ],
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "dataset-split-names": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-split-names",
        ],
        "job_runner_version": 3,
        "difficulty": 20,
    },
    "split-descriptive-statistics": {
        "input_type": "split",
        "triggered_by": "config-parquet-metadata",
        "job_runner_version": 3,
        "difficulty": 70,
        "bonus_difficulty_if_dataset_is_big": 20,
    },
    "split-is-valid": {
        "input_type": "split",
        "triggered_by": [
            "config-size",
            "split-first-rows",
            "split-duckdb-index",
            "split-descriptive-statistics",
        ],
        "job_runner_version": 4,
        "difficulty": 20,
    },
    "config-is-valid": {
        "input_type": "config",
        "triggered_by": [
            "config-split-names",  # required in case the config has no splits (error in previous step)
            "split-is-valid",
        ],
        "job_runner_version": 3,
        "difficulty": 20,
    },
    "dataset-is-valid": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-is-valid",
        ],
        "job_runner_version": 7,
        "difficulty": 20,
    },
    "split-image-url-columns": {
        "input_type": "split",
        "triggered_by": "split-first-rows",
        "job_runner_version": 1,
        "difficulty": 40,
    },
    "split-opt-in-out-urls-scan": {
        "input_type": "split",
        "triggered_by": "split-image-url-columns",
        "job_runner_version": 4,
        "difficulty": 70,
    },
    "split-opt-in-out-urls-count": {
        "input_type": "split",
        "triggered_by": "split-opt-in-out-urls-scan",
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "config-opt-in-out-urls-count": {
        "input_type": "config",
        "triggered_by": [
            "config-split-names",  # required in case the config has no splits (error in previous step)
            "split-opt-in-out-urls-count",
        ],
        "job_runner_version": 3,
        "difficulty": 20,
    },
    "dataset-opt-in-out-urls-count": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-opt-in-out-urls-count",
        ],
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "split-duckdb-index": {
        "input_type": "split",
        "triggered_by": "config-parquet-metadata",
        "job_runner_version": 2,
        "difficulty": 70,
        "bonus_difficulty_if_dataset_is_big": 20,
    },
    "config-duckdb-index-size": {
        "input_type": "config",
        "triggered_by": [
            "config-split-names",  # required in case the config has no splits (error in previous step)
            "split-duckdb-index",
        ],
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "dataset-duckdb-index-size": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-config-names",  # required in case the dataset has no configs (error in previous step)
            "config-duckdb-index-size",
        ],
        "job_runner_version": 1,
        "difficulty": 20,
    },
    "dataset-hub-cache": {
        "input_type": "dataset",
        "triggered_by": [
            "dataset-is-valid",
            "dataset-size",
            "dataset-compatible-libraries",
            "dataset-modalities",
        ],
        "job_runner_version": 2,
        "difficulty": 20,
    },
    "dataset-compatible-libraries": {
        "input_type": "dataset",
        "triggered_by": "dataset-info",
        "job_runner_version": 5,
        "difficulty": 20,
    },
    "dataset-modalities": {
        "input_type": "dataset",
        "triggered_by": "dataset-info",
        "job_runner_version": 1,
        "difficulty": 20,
    },
    "dataset-croissant-crumbs": {
        "input_type": "dataset",
        "triggered_by": "dataset-info",
        "job_runner_version": 1,
        "difficulty": 20,
    },
}

# global variable
processing_graph = ProcessingGraph(specification=specification)

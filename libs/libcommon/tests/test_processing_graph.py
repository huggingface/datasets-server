# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.processing_graph import ProcessingGraph, ProcessingGraphSpecification, ProcessingStep, processing_graph


def assert_lists_are_equal(a: list[ProcessingStep], b: list[str]) -> None:
    assert sorted(processing_step.name for processing_step in a) == sorted(b)


def assert_step(
    graph: ProcessingGraph,
    processing_step_name: str,
    children: list[str],
    parents: list[str],
    ancestors: list[str],
) -> None:
    assert_lists_are_equal(graph.get_children(processing_step_name), children)
    assert_lists_are_equal(graph.get_parents(processing_step_name), parents)
    assert_lists_are_equal(graph.get_ancestors(processing_step_name), ancestors)


def test_graph() -> None:
    a = "step_a"
    b = "step_b"
    c = "step_c"
    d = "step_d"
    e = "step_e"
    f = "step_f"
    specification: ProcessingGraphSpecification = {
        a: {"input_type": "dataset", "job_runner_version": 1},
        b: {"input_type": "dataset", "job_runner_version": 1},
        c: {"input_type": "dataset", "triggered_by": a, "job_runner_version": 1},
        d: {"input_type": "dataset", "triggered_by": [a, c], "job_runner_version": 1},
        e: {"input_type": "dataset", "triggered_by": [c], "job_runner_version": 1},
        f: {"input_type": "dataset", "triggered_by": [a, b], "job_runner_version": 1},
    }
    graph = ProcessingGraph(specification)

    assert_step(graph, a, children=[c, d, f], parents=[], ancestors=[])
    assert_step(graph, b, children=[f], parents=[], ancestors=[])
    assert_step(graph, c, children=[d, e], parents=[a], ancestors=[a])
    assert_step(graph, d, children=[], parents=[a, c], ancestors=[a, c])
    assert_step(graph, e, children=[], parents=[c], ancestors=[a, c])
    assert_step(graph, f, children=[], parents=[a, b], ancestors=[a, b])


@pytest.mark.parametrize(
    "processing_step_name,children,parents,ancestors",
    [
        (
            "dataset-config-names",
            [
                "config-split-names",
                "config-parquet-and-info",
                "dataset-duckdb-index-size",
                "dataset-opt-in-out-urls-count",
                "dataset-split-names",
                "dataset-parquet",
                "dataset-info",
                "dataset-size",
                "dataset-is-valid",
            ],
            [],
            [],
        ),
        (
            "config-parquet-and-info",
            [
                "config-parquet",
                "config-info",
                "config-size",
            ],
            ["dataset-config-names"],
            ["dataset-config-names"],
        ),
        (
            "config-split-names",
            [
                "config-duckdb-index-size",
                "config-opt-in-out-urls-count",
                "split-first-rows",
                "dataset-split-names",
                "config-is-valid",
            ],
            ["dataset-config-names", "config-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info"],
        ),
        (
            "dataset-split-names",
            [],
            [
                "dataset-config-names",
                "config-split-names",
            ],
            [
                "dataset-config-names",
                "config-parquet-and-info",
                "config-info",
                "config-split-names",
            ],
        ),
        (
            "split-first-rows",
            ["split-is-valid", "split-image-url-columns"],
            ["config-parquet-metadata", "config-split-names"],
            [
                "config-parquet",
                "dataset-config-names",
                "config-split-names",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "config-info",
            ],
        ),
        (
            "config-parquet",
            ["config-parquet-metadata", "dataset-parquet"],
            ["config-parquet-and-info"],
            ["dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "config-parquet-metadata",
            ["split-first-rows", "split-duckdb-index", "split-descriptive-statistics"],
            ["config-parquet"],
            ["dataset-config-names", "config-parquet-and-info", "config-parquet"],
        ),
        (
            "dataset-parquet",
            [],
            ["dataset-config-names", "config-parquet"],
            ["dataset-config-names", "config-parquet-and-info", "config-parquet"],
        ),
        (
            "config-info",
            ["dataset-info", "config-split-names"],
            ["config-parquet-and-info"],
            ["dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "dataset-info",
            ["dataset-compatible-libraries", "dataset-modalities", "dataset-croissant-crumbs"],
            ["dataset-config-names", "config-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info"],
        ),
        (
            "config-size",
            ["split-is-valid", "dataset-size"],
            ["config-parquet-and-info"],
            ["dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "dataset-size",
            ["dataset-hub-cache"],
            ["dataset-config-names", "config-size"],
            ["dataset-config-names", "config-parquet-and-info", "config-size"],
        ),
        (
            "dataset-compatible-libraries",
            ["dataset-hub-cache"],
            ["dataset-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info", "dataset-info"],
        ),
        (
            "dataset-modalities",
            ["dataset-hub-cache"],
            ["dataset-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info", "dataset-info"],
        ),
        (
            "dataset-is-valid",
            ["dataset-hub-cache"],
            [
                "config-is-valid",
                "dataset-config-names",
            ],
            [
                "dataset-config-names",
                "config-parquet-and-info",
                "config-info",
                "config-parquet",
                "config-size",
                "config-split-names",
                "config-parquet-metadata",
                "split-first-rows",
                "config-is-valid",
                "split-is-valid",
                "split-duckdb-index",
                "split-descriptive-statistics",
            ],
        ),
        (
            "split-image-url-columns",
            ["split-opt-in-out-urls-scan"],
            ["split-first-rows"],
            [
                "dataset-config-names",
                "config-split-names",
                "config-info",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "split-first-rows",
                "config-parquet",
            ],
        ),
        (
            "split-opt-in-out-urls-scan",
            ["split-opt-in-out-urls-count"],
            ["split-image-url-columns"],
            [
                "dataset-config-names",
                "config-split-names",
                "config-info",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "split-first-rows",
                "config-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "split-opt-in-out-urls-count",
            ["config-opt-in-out-urls-count"],
            ["split-opt-in-out-urls-scan"],
            [
                "dataset-config-names",
                "config-split-names",
                "split-first-rows",
                "config-info",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "split-opt-in-out-urls-scan",
                "config-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "config-opt-in-out-urls-count",
            ["dataset-opt-in-out-urls-count"],
            ["split-opt-in-out-urls-count", "config-split-names"],
            [
                "dataset-config-names",
                "config-split-names",
                "split-first-rows",
                "config-info",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "split-opt-in-out-urls-count",
                "split-opt-in-out-urls-scan",
                "config-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "dataset-opt-in-out-urls-count",
            [],
            ["config-opt-in-out-urls-count", "dataset-config-names"],
            [
                "dataset-config-names",
                "config-split-names",
                "split-first-rows",
                "config-info",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "config-opt-in-out-urls-count",
                "split-opt-in-out-urls-count",
                "split-opt-in-out-urls-scan",
                "config-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "split-duckdb-index",
            ["config-duckdb-index-size", "split-is-valid"],
            ["config-parquet-metadata"],
            [
                "config-parquet",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "dataset-config-names",
            ],
        ),
        (
            "config-duckdb-index-size",
            ["dataset-duckdb-index-size"],
            ["split-duckdb-index", "config-split-names"],
            [
                "config-split-names",
                "config-parquet",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "config-info",
                "dataset-config-names",
                "split-duckdb-index",
            ],
        ),
        (
            "dataset-duckdb-index-size",
            [],
            ["config-duckdb-index-size", "dataset-config-names"],
            [
                "config-duckdb-index-size",
                "config-split-names",
                "config-parquet",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "config-info",
                "dataset-config-names",
                "split-duckdb-index",
            ],
        ),
        (
            "split-descriptive-statistics",
            ["split-is-valid"],
            ["config-parquet-metadata"],
            [
                "config-parquet-metadata",
                "config-parquet",
                "config-parquet-and-info",
                "dataset-config-names",
            ],
        ),
        (
            "dataset-hub-cache",
            [],
            ["dataset-is-valid", "dataset-size", "dataset-compatible-libraries", "dataset-modalities"],
            [
                "config-info",
                "config-is-valid",
                "config-parquet",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "config-size",
                "config-split-names",
                "dataset-config-names",
                "dataset-info",
                "dataset-is-valid",
                "dataset-compatible-libraries",
                "dataset-modalities",
                "dataset-size",
                "split-duckdb-index",
                "split-first-rows",
                "split-is-valid",
                "split-descriptive-statistics",
            ],
        ),
        (
            "split-is-valid",
            ["config-is-valid"],
            ["split-first-rows", "split-duckdb-index", "split-descriptive-statistics", "config-size"],
            [
                "config-parquet-metadata",
                "config-split-names",
                "config-parquet",
                "config-parquet-and-info",
                "config-info",
                "config-size",
                "dataset-config-names",
                "split-duckdb-index",
                "split-first-rows",
                "split-descriptive-statistics",
            ],
        ),
        (
            "config-is-valid",
            ["dataset-is-valid"],
            ["config-split-names", "split-is-valid"],
            [
                "config-split-names",
                "config-parquet",
                "config-parquet-and-info",
                "config-parquet-metadata",
                "config-info",
                "config-size",
                "dataset-config-names",
                "split-duckdb-index",
                "split-first-rows",
                "split-descriptive-statistics",
                "split-is-valid",
            ],
        ),
        (
            "dataset-croissant-crumbs",
            [],
            ["dataset-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info", "dataset-info"],
        ),
    ],
)
def test_default_graph_steps(
    processing_step_name: str, children: list[str], parents: list[str], ancestors: list[str]
) -> None:
    assert_step(processing_graph, processing_step_name, children=children, parents=parents, ancestors=ancestors)


def test_default_graph_first_steps() -> None:
    roots = ["dataset-config-names"]
    assert_lists_are_equal(processing_graph.get_first_processing_steps(), roots)

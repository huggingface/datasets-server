# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
from http import HTTPStatus
from typing import Dict, Optional

import pytest
from libcache.simple_cache import _clean_cache_database, upsert_response
from libcommon.processing_steps import (
    Parameters,
    ProcessingStep,
    first_rows_step,
    parquet_step,
    splits_step,
)
from libqueue.queue import Queue, _clean_queue_database
from pytest_httpserver import HTTPServer
from starlette.testclient import TestClient

from api.app import create_app
from api.config import AppConfig

from .utils import auth_callback


@pytest.fixture(scope="module")
def client(monkeypatch_session: pytest.MonkeyPatch) -> TestClient:
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def clean_mongo_databases(app_config: AppConfig) -> None:
    _clean_cache_database()
    _clean_queue_database()


def test_cors(client: TestClient) -> None:
    origin = "http://localhost:3000"
    method = "GET"
    header = "X-Requested-With"
    response = client.options(
        f"{splits_step.endpoint}?dataset=dataset1",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": header,
        },
    )
    assert response.status_code == 200
    assert (
        origin in [o.strip() for o in response.headers["Access-Control-Allow-Origin"].split(",")]
        or response.headers["Access-Control-Allow-Origin"] == "*"
    )
    assert (
        header in [o.strip() for o in response.headers["Access-Control-Allow-Headers"].split(",")]
        or response.headers["Access-Control-Expose-Headers"] == "*"
    )
    assert (
        method in [o.strip() for o in response.headers["Access-Control-Allow-Methods"].split(",")]
        or response.headers["Access-Control-Expose-Headers"] == "*"
    )
    assert response.headers["Access-Control-Allow-Credentials"] == "true"


def test_get_valid_datasets(client: TestClient) -> None:
    response = client.get("/valid")
    assert response.status_code == 200
    json = response.json()
    assert "valid" in json


# caveat: the returned status codes don't simulate the reality
# they're just used to check every case
@pytest.mark.parametrize(
    "headers,status_code,error_code",
    [
        ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
        ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
        ({}, 200, None),
    ],
)
def test_is_valid_auth(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    headers: Dict[str, str],
    status_code: int,
    error_code: Optional[str],
) -> None:
    dataset = "dataset-which-does-not-exist"
    httpserver.expect_request(hf_auth_path % dataset, headers=headers).respond_with_handler(auth_callback)
    response = client.get(f"/is-valid?dataset={dataset}", headers=headers)
    assert response.status_code == status_code
    assert response.headers.get("X-Error-Code") == error_code


def test_get_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


def test_get_splits(client: TestClient) -> None:
    # missing parameter
    response = client.get(splits_step.endpoint)
    assert response.status_code == 422
    # empty parameter
    response = client.get(f"{splits_step.endpoint}?dataset=")
    assert response.status_code == 422


# caveat: the returned status codes don't simulate the reality
# they're just used to check every case
@pytest.mark.parametrize(
    "headers,status_code,error_code",
    [
        ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
        ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
        ({}, 500, "ResponseNotReady"),
    ],
)
def test_splits_auth(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    headers: Dict[str, str],
    status_code: int,
    error_code: str,
) -> None:
    dataset = "dataset-which-does-not-exist"
    httpserver.expect_request(hf_auth_path % dataset, headers=headers).respond_with_handler(auth_callback)
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({}), headers={"X-Error-Code": "RepoNotFound"}
    )
    response = client.get(f"{splits_step.endpoint}?dataset={dataset}", headers=headers)
    assert response.status_code == status_code, f"{response.headers}, {response.json()}"
    assert response.headers.get("X-Error-Code") == error_code


@pytest.mark.parametrize(
    "dataset,config,split",
    [
        (None, None, None),
        ("a", None, None),
        ("a", "b", None),
        ("a", "b", ""),
    ],
)
def test_get_first_rows_missing_parameter(
    client: TestClient, dataset: Optional[str], config: Optional[str], split: Optional[str]
) -> None:
    response = client.get(first_rows_step.endpoint, params={"dataset": dataset, "config": config, "split": split})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "processing_step,exists,is_private,expected_error_code",
    [
        (splits_step, False, None, "ExternalAuthenticatedError"),
        (splits_step, True, True, "ResponseNotFound"),
        (splits_step, True, False, "ResponseNotReady"),
        (parquet_step, False, None, "ExternalAuthenticatedError"),
        (parquet_step, True, True, "ResponseNotFound"),
        (parquet_step, True, False, "ResponseNotReady"),
        (first_rows_step, False, None, "ExternalAuthenticatedError"),
        (first_rows_step, True, True, "ResponseNotFound"),
        (first_rows_step, True, False, "ResponseNotReady"),
    ],
)
def test_cache_refreshing(
    client: TestClient,
    httpserver: HTTPServer,
    hf_auth_path: str,
    processing_step: ProcessingStep,
    exists: bool,
    is_private: Optional[bool],
    expected_error_code: str,
) -> None:
    dataset = "dataset-to-be-processed"
    config = None if processing_step.parameters == Parameters.DATASET else "config"
    split = None if processing_step.parameters == Parameters.DATASET else "split"
    httpserver.expect_request(hf_auth_path % dataset).respond_with_data(status=200 if exists else 404)
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({"private": is_private}), headers={} if exists else {"X-Error-Code": "RepoNotFound"}
    )

    response = client.get(processing_step.endpoint, params={"dataset": dataset, "config": config, "split": split})
    assert response.headers["X-Error-Code"] == expected_error_code

    if expected_error_code == "ResponseNotReady":
        # a subsequent request should return the same error code
        response = client.get(processing_step.endpoint, params={"dataset": dataset, "config": config, "split": split})
        assert response.headers["X-Error-Code"] == expected_error_code

        # simulate the worker
        upsert_response(
            kind=processing_step.cache_kind,
            dataset=dataset,
            config=config,
            split=split,
            content={"key": "value"},
            http_status=HTTPStatus.OK,
        )
        response = client.get(processing_step.endpoint, params={"dataset": dataset, "config": config, "split": split})
        assert response.json()["key"] == "value"
        assert response.status_code == 200


def test_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    lines = text.split("\n")
    metrics = {line.split(" ")[0]: float(line.split(" ")[1]) for line in lines if line and line[0] != "#"}

    # the middleware should have recorded the request
    name = 'starlette_requests_total{method="GET",path_template="/metrics"}'
    assert name in metrics, metrics
    assert metrics[name] > 0, metrics


@pytest.mark.parametrize(
    "payload,exists_on_the_hub,expected_status,expected_is_updated",
    [
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, True, 200, True),
        (
            {
                "event": "move",
                "movedTo": "webhook-test",
                "repo": {"type": "dataset", "name": "previous-name", "gitalyUid": "123"},
            },
            True,
            200,
            True,
        ),
        (
            {"event": "doesnotexist", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}},
            True,
            400,
            False,
        ),
        (
            {"event": "add", "repo": {"type": "dataset", "name": "webhook-test"}},
            True,
            200,
            True,
        ),
        ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, False, 400, False),
    ],
)
def test_webhook(
    client: TestClient,
    httpserver: HTTPServer,
    payload: Dict,
    exists_on_the_hub: bool,
    expected_status: int,
    expected_is_updated: bool,
) -> None:
    dataset = "webhook-test"
    headers = None if exists_on_the_hub else {"X-Error-Code": "RepoNotFound"}
    status = 200 if exists_on_the_hub else 404
    httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
        json.dumps({"private": False}), headers=headers, status=status
    )
    response = client.post("/webhook", json=payload)
    assert response.status_code == expected_status, response.text
    assert Queue(type=splits_step.job_type).is_job_in_process(dataset=dataset) is expected_is_updated
    assert Queue(type=parquet_step.job_type).is_job_in_process(dataset=dataset) is expected_is_updated
    assert Queue(type=first_rows_step.job_type).is_job_in_process(dataset=dataset) is False

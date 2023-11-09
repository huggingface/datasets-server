# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from .utils import SEARCH_URL, poll


def test_healthcheck() -> None:
    response = poll("/healthcheck", expected_code=200, url=SEARCH_URL)
    assert response.status_code == 200, f"{response.status_code} - {response.text}"
    assert "ok" in response.text, response.text

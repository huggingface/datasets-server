# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from typing import Any

import pytest

from libcommon.cloudfront import URLSigner
from libcommon.storage_client import StorageClient

BASE_URL = "https://baseurl/assets"


class FakeUrlSigner(URLSigner):
    def __init__(self) -> None:
        self.counter = 0

    def sign_url(self, url: str) -> str:
        self.counter += 1
        return url


URL = f"{BASE_URL}/file.txt"
STRING_WITH_URL_AT_THE_START = f"{URL} and some other text"
OBJECT_WITH_URL = {"key": URL, "key2": URL}
ARRAY_WITH_URL = [URL, URL]
NESTED_OBJECT_WITH_URL = {"key": {"key2": URL}, "key2": URL, "key3": {"key4": [URL]}}

STRING_WITH_NO_URL = "string with no url"
STRING_WITH_URL_INSIDE = f"string with url inside {URL}"
OBJECT_WITH_NO_URL = {"key": "value"}
ARRAY_WITH_NO_URL = ["value1", "value2"]


@pytest.mark.parametrize(
    "content,num_replaced_urls",
    [
        (URL, 1),
        (STRING_WITH_URL_AT_THE_START, 1),
        (OBJECT_WITH_URL, 2),
        (ARRAY_WITH_URL, 2),
        (NESTED_OBJECT_WITH_URL, 3),
        # ^ not expected, hopefully will not happen
        # below: will not sign any URL
        (None, 0),
        (1, 0),
        (STRING_WITH_NO_URL, 0),
        (STRING_WITH_URL_INSIDE, 0),
        (OBJECT_WITH_NO_URL, 0),
    ],
)
def test_sign_urls_in_obj(content: Any, num_replaced_urls: int) -> None:
    url_signer = FakeUrlSigner()
    storage_client = StorageClient(protocol="file", storage_root="/tmp", base_url=BASE_URL, url_signer=url_signer)
    storage_client.sign_urls_in_obj(content)
    assert url_signer.counter == num_replaced_urls

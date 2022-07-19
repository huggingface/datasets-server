import pytest
from libcache.cache import DbDataset
from libcache.cache import clean_database as clean_cache_database
from libcache.cache import connect_to_cache, get_rows_response
from libcache.cache import get_splits_response as old_get_splits_response
from libcache.simple_cache import (
    HTTPStatus,
    get_first_rows_response,
    get_splits_response,
)
from libqueue.queue import clean_database as clean_queue_database
from libqueue.queue import connect_to_queue
from libutils.exceptions import Status400Error

from worker.refresh import (
    refresh_dataset,
    refresh_first_rows,
    refresh_split,
    refresh_splits,
)

from ._utils import (
    ASSETS_BASE_URL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
)


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    clean_cache_database()
    clean_queue_database()


def test_doesnotexist() -> None:
    dataset_name = "doesnotexist"
    with pytest.raises(Status400Error):
        refresh_dataset(dataset_name)
    # TODO: don't use internals of the cache database?
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status.value == "error"

    assert refresh_splits(dataset_name) == HTTPStatus.BAD_REQUEST
    response, http_status = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.BAD_REQUEST
    assert response["status_code"] == 400
    assert response["exception"] == "Status400Error"


def test_e2e_examples() -> None:
    # see https://github.com/huggingface/datasets-server/issues/78
    dataset_name = "Check/region_1"
    refresh_dataset(dataset_name)
    # TODO: don't use internals of the cache database?
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status.value == "valid"
    splits_response, error, status_code = old_get_splits_response(dataset_name)
    assert status_code == 200
    assert error is None
    assert splits_response is not None
    assert "splits" in splits_response
    assert len(splits_response["splits"]) == 1

    assert refresh_splits(dataset_name) == HTTPStatus.OK
    response, _ = get_splits_response(dataset_name)
    assert len(response["splits"]) == 1
    assert response["splits"][0]["num_bytes"] is None
    assert response["splits"][0]["num_examples"] is None

    dataset_name = "acronym_identification"
    assert refresh_splits(dataset_name) == HTTPStatus.OK
    response, _ = get_splits_response(dataset_name)
    assert len(response["splits"]) == 3
    assert response["splits"][0]["num_bytes"] == 7792803
    assert response["splits"][0]["num_examples"] == 14006


def test_large_document() -> None:
    # see https://github.com/huggingface/datasets-server/issues/89
    dataset_name = "SaulLu/Natural_Questions_HTML"
    refresh_dataset(dataset_name)
    retrieved = DbDataset.objects(dataset_name=dataset_name).get()
    assert retrieved.status.value == "valid"

    assert refresh_splits(dataset_name) == HTTPStatus.OK
    _, http_status = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.OK


def test_column_order() -> None:
    refresh_split("acronym_identification", "default", "train")
    rows_response, error, status_code = get_rows_response("acronym_identification", "default", "train")
    assert status_code == 200
    assert error is None
    assert rows_response is not None
    assert "columns" in rows_response
    assert rows_response["columns"][0]["column"]["name"] == "id"
    assert rows_response["columns"][1]["column"]["name"] == "tokens"
    assert rows_response["columns"][2]["column"]["name"] == "labels"


def test_first_rows() -> None:
    http_status = refresh_first_rows("common_voice", "tr", "train", ASSETS_BASE_URL)
    response, cached_http_status = get_first_rows_response("common_voice", "tr", "train")
    assert http_status == HTTPStatus.OK
    assert cached_http_status == HTTPStatus.OK

    assert response["features"][0]["idx"] == 0
    assert response["features"][0]["name"] == "client_id"
    assert response["features"][0]["type"]["_type"] == "Value"
    assert response["features"][0]["type"]["dtype"] == "string"

    assert response["features"][2]["name"] == "audio"
    assert response["features"][2]["type"]["_type"] == "Audio"
    assert response["features"][2]["type"]["sampling_rate"] == 48000

    assert response["rows"][0]["row_idx"] == 0
    assert response["rows"][0]["row"]["client_id"].startswith("54fc2d015c27a057b")
    assert response["rows"][0]["row"]["audio"] == [
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.mp3", "type": "audio/mpeg"},
        {"src": f"{ASSETS_BASE_URL}/common_voice/--/tr/train/0/audio/audio.wav", "type": "audio/wav"},
    ]

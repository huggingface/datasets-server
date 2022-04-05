from PIL import Image  # type: ignore

from datasets_preview_backend.config import ROWS_MAX_NUMBER
from datasets_preview_backend.models.row import get_rows


# get_rows
def test_get_rows() -> None:
    rows = get_rows("acronym_identification", "default", "train")
    assert len(rows) == ROWS_MAX_NUMBER
    assert rows[0]["tokens"][0] == "What"


def test_class_label() -> None:
    rows = get_rows("glue", "cola", "train")
    assert rows[0]["label"] == 1


def test_mnist() -> None:
    rows = get_rows("mnist", "mnist", "train")
    assert len(rows) == ROWS_MAX_NUMBER
    assert isinstance(rows[0]["image"], Image.Image)


def test_cifar() -> None:
    rows = get_rows("cifar10", "plain_text", "train")
    assert len(rows) == ROWS_MAX_NUMBER
    assert isinstance(rows[0]["img"], Image.Image)


def test_iter_archive() -> None:
    rows = get_rows("food101", "default", "train")
    assert len(rows) == ROWS_MAX_NUMBER
    assert isinstance(rows[0]["image"], Image.Image)


def test_dl_1_suffix() -> None:
    # see https://github.com/huggingface/datasets/pull/2843
    rows = get_rows("discovery", "discovery", "train")
    assert len(rows) == ROWS_MAX_NUMBER


def test_txt_zip() -> None:
    # see https://github.com/huggingface/datasets/pull/2856
    rows = get_rows("bianet", "en_to_ku", "train")
    assert len(rows) == ROWS_MAX_NUMBER


def test_pathlib() -> None:
    # see https://github.com/huggingface/datasets/issues/2866
    rows = get_rows("counter", "counter", "train")
    assert len(rows) == ROWS_MAX_NUMBER


def test_community_with_no_config() -> None:
    rows = get_rows("Check/region_1", "Check--region_1", "train")
    # it's not correct: here this is the number of splits, not the number of rows
    assert len(rows) == 2
    # see https://github.com/huggingface/datasets-preview-backend/issues/78
    get_rows("Check/region_1", "Check--region_1", "train")


def test_audio_dataset() -> None:
    rows = get_rows("abidlabs/test-audio-1", "test", "train")
    assert len(rows) == 1
    assert rows[0]["Output"]["sampling_rate"] == 48000


def test_libsndfile() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/194
    rows = get_rows("polinaeterna/ml_spoken_words", "ar_opus", "train")
    assert len(rows) == ROWS_MAX_NUMBER
    assert rows[0]["audio"]["sampling_rate"] == 48000

    rows = get_rows("polinaeterna/ml_spoken_words", "ar_wav", "train")
    assert len(rows) == ROWS_MAX_NUMBER
    assert rows[0]["audio"]["sampling_rate"] == 16000

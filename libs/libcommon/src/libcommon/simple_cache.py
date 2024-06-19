# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import types
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from http import HTTPStatus
from typing import Any, Generic, NamedTuple, Optional, TypedDict, TypeVar, overload

import pandas as pd
import pyarrow as pa
from bson import CodecOptions, ObjectId
from bson.codec_options import TypeEncoder, TypeRegistry  # type: ignore[attr-defined]
from bson.errors import InvalidId
from mongoengine import Document
from mongoengine.errors import DoesNotExist
from mongoengine.fields import (
    DateTimeField,
    DictField,
    EnumField,
    FloatField,
    IntField,
    ObjectIdField,
    StringField,
)
from mongoengine.queryset.queryset import QuerySet
from pymongoarrow.api import Schema, find_pandas_all

from libcommon.constants import (
    CACHE_COLLECTION_RESPONSES,
    CACHE_METRICS_COLLECTION,
    CACHE_MONGOENGINE_ALIAS,
    ERROR_CODES_TO_RETRY,
)
from libcommon.dtos import JobParams
from libcommon.utils import get_datetime


class DateCodec(TypeEncoder):  # type: ignore[misc]
    """To be able to save datetime.date objects like datetime.datetime objects in mongo"""

    python_type = date  # the Python type acted upon by this type codec
    transform_python = str  # convert date objects to strings in mongo


class TimeCodec(TypeEncoder):  # type: ignore[misc]
    """To be able to save datetime.time objects as strings in mongo"""

    python_type = time  # the Python type acted upon by this type codec
    transform_python = str  # convert time objects to strings in mongo


class TimedeltaCodec(TypeEncoder):  # type: ignore[misc]
    """To be able to save datetime.timedelta objects as strings in mongo"""

    python_type = timedelta  # the Python type acted upon by this type codec
    transform_python = str  # convert timedelta objects to strings in mongo


class DecimalCodec(TypeEncoder):  # type: ignore[misc]
    """To be able to save decimal.Decimal objects as strings in mongo"""

    python_type = Decimal  # the Python type acted upon by this type codec
    transform_python = str  # convert decimal objects to strings in mongo


type_registry = TypeRegistry([DateCodec(), TimeCodec(), TimedeltaCodec(), DecimalCodec()])


# START monkey patching ### hack ###
# see https://github.com/sbdchd/mongo-types#install
U = TypeVar("U", bound=Document)


def no_op(self, _):  # type: ignore
    return self


QuerySet.__class_getitem__ = types.MethodType(no_op, QuerySet)


class QuerySetManager(Generic[U]):
    def __get__(self, instance: object, cls: type[U]) -> QuerySet[U]:
        codec_options = CodecOptions(type_registry=type_registry)  # type: ignore[call-arg]
        cls._collection = cls._get_db().get_collection(  # type: ignore[attr-defined]
            cls._get_collection_name(), codec_options=codec_options
        )
        return QuerySet(cls, cls._get_collection())


# END monkey patching ### hack ###


class SplitFullName(NamedTuple):
    """A split full name is a tuple of (dataset, config, split)."""

    dataset: str
    config: Optional[str]
    split: Optional[str]


PA_SCHEMA = Schema(
    {
        "kind": pa.string(),
        "dataset": pa.string(),
        "config": pa.string(),
        "split": pa.string(),
        "http_status": pa.int32(),
        "error_code": pa.string(),
        "dataset_git_revision": pa.string(),
        "job_runner_version": pa.int32(),
        "progress": pa.float64(),
        "updated_at": pa.timestamp("ms"),
        "failed_runs": pa.int32(),
    }
)


# cache of any job
class CachedResponseDocument(Document):
    """A response computed for a job, cached in the mongoDB database

    Args:
        kind (`str`): The kind of the cached response, identifies the job type
        dataset (`str`): The requested dataset.
        config (`str`, *optional*): The requested config, if any.
        split (`str`, *optional*): The requested split, if any.
        http_status (`HTTPStatus`): The HTTP status code.
        error_code (`str`, *optional*): The error code, if any.
        content (`dict`): The content of the cached response. Can be an error or a valid content.
        details (`dict`, *optional*): Additional details, eg. a detailed error that we don't want to send as a response.
        started_at (`datetime`, *optional*): When the cache entry's last job started.
        updated_at (`datetime`): When the cache entry has been last updated.
        job_runner_version (`int`): The version of the job runner that cached the response.
        failed_runs (`int`): The number of failed_runs to get cached result.
        dataset_git_revision (`str`): The commit (of the git dataset repo) used to generate the response.
        progress (`float`): Progress percentage (between 0. and 1.) if the result is not complete yet.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)

    kind = StringField(required=True, unique_with=["dataset", "config", "split"])
    dataset = StringField(required=True)
    config = StringField()
    split = StringField()

    http_status = EnumField(HTTPStatus, required=True)
    error_code = StringField()
    content = DictField(required=True)
    dataset_git_revision = StringField(required=True)
    progress = FloatField(min_value=0.0, max_value=1.0)
    job_runner_version = IntField()
    failed_runs = IntField(default=0)
    details = DictField()
    started_at = DateTimeField()
    updated_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": CACHE_COLLECTION_RESPONSES,
        "db_alias": CACHE_MONGOENGINE_ALIAS,
        "indexes": [
            ("kind", "dataset", "config", "split"),
            ("dataset", "kind", "http_status"),
            ("kind", "http_status", "error_code"),
            ("kind", "http_status", "_id"),
            ("kind", "_id"),  # < recommended by Atlas
            ("details.cause_exception", "error_code", "details.copied_from_artifact"),  # < recommended by Atlas
            ("error_code", "kind", "details.copied_from_artifact"),  # < recommended by Atlas
            ("http_status", "error_code", "kind", "updated_at"),  # < recommended by Atlas
        ],
    }
    objects = QuerySetManager["CachedResponseDocument"]()

    @classmethod
    def fetch_as_df(cls, query: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch documents matching the query as a Pandas Dataframe.
        """
        query = query if query is not None else {}
        collection = cls._get_collection()
        return find_pandas_all(collection, query, schema=PA_SCHEMA)  # type: ignore


DEFAULT_INCREASE_AMOUNT = 1
DEFAULT_DECREASE_AMOUNT = -1


class CacheTotalMetricDocument(Document):
    """Cache total metric in the mongoDB database, used to compute prometheus metrics.

    Args:
        kind (`str`): kind name
        http_status (`int`): cache http_status
        error_code (`str`): error code name
        total (`int`): total of jobs
        created_at (`datetime`): when the metric has been created.
    """

    id = ObjectIdField(db_field="_id", primary_key=True, default=ObjectId)
    kind = StringField(required=True)
    http_status = IntField(required=True)
    error_code = StringField()
    total = IntField(required=True, default=0)
    created_at = DateTimeField(default=get_datetime)

    meta = {
        "collection": CACHE_METRICS_COLLECTION,
        "db_alias": CACHE_MONGOENGINE_ALIAS,
        "indexes": [
            {
                "fields": ["kind", "http_status", "error_code"],
                "unique": True,
            }
        ],
    }
    objects = QuerySetManager["CacheTotalMetricDocument"]()


# Fix issue with mongoengine: https://github.com/MongoEngine/mongoengine/issues/1242#issuecomment-810501601
# mongoengine automatically sets "config" and "splits" as required fields, because they are listed in the unique_with
# field of the "kind" field. But it's an error, since unique indexes (which are used to enforce unique_with) accept
# null values, see https://www.mongodb.com/docs/v5.0/core/index-unique/#unique-index-and-missing-field.
CachedResponseDocument.config.required = False  # type: ignore
CachedResponseDocument.split.required = False  # type: ignore


class CachedArtifactNotFoundError(Exception):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]

    def __init__(
        self,
        kind: str,
        dataset: str,
        config: Optional[str],
        split: Optional[str],
    ):
        super().__init__(f"Cache entry does not exist: {kind=} {dataset=} {config=} {split=}")
        self.kind = kind
        self.dataset = dataset
        self.config = config
        self.split = split


def _update_metrics(kind: str, http_status: HTTPStatus, increase_by: int, error_code: Optional[str] = None) -> None:
    CacheTotalMetricDocument.objects(kind=kind, http_status=http_status, error_code=error_code).upsert_one(
        inc__total=increase_by
    )


def increase_metric(kind: str, http_status: HTTPStatus, error_code: Optional[str] = None) -> None:
    _update_metrics(kind=kind, http_status=http_status, error_code=error_code, increase_by=DEFAULT_INCREASE_AMOUNT)


def decrease_metric(kind: str, http_status: HTTPStatus, error_code: Optional[str] = None) -> None:
    _update_metrics(kind=kind, http_status=http_status, error_code=error_code, increase_by=DEFAULT_DECREASE_AMOUNT)


def decrease_metric_for_artifact(kind: str, dataset: str, config: Optional[str], split: Optional[str]) -> None:
    try:
        existing_cache = CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split).get()
    except DoesNotExist:
        return
    decrease_metric(kind=kind, http_status=existing_cache.http_status, error_code=existing_cache.error_code)


# Note: we let the exceptions throw (ie DocumentTooLarge): it's the responsibility of the caller to manage them
def upsert_response(
    kind: str,
    dataset: str,
    dataset_git_revision: str,
    content: Mapping[str, Any],
    http_status: HTTPStatus,
    config: Optional[str] = None,
    split: Optional[str] = None,
    error_code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    job_runner_version: Optional[int] = None,
    progress: Optional[float] = None,
    started_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
    failed_runs: int = 0,
) -> None:
    decrease_metric_for_artifact(kind=kind, dataset=dataset, config=config, split=split)
    CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split).upsert_one(
        content=content,
        http_status=http_status,
        error_code=error_code,
        details=details,
        dataset_git_revision=dataset_git_revision,
        progress=progress,
        started_at=started_at,
        updated_at=updated_at or get_datetime(),
        job_runner_version=job_runner_version,
        failed_runs=failed_runs,
    )
    increase_metric(kind=kind, http_status=http_status, error_code=error_code)


def upsert_response_params(
    kind: str,
    job_params: JobParams,
    content: Mapping[str, Any],
    http_status: HTTPStatus,
    error_code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    job_runner_version: Optional[int] = None,
    progress: Optional[float] = None,
    started_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
    failed_runs: int = 0,
) -> None:
    dataset, config, split, revision = (
        job_params["dataset"],
        job_params["config"],
        job_params["split"],
        job_params["revision"],
    )
    upsert_response(
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        dataset_git_revision=revision,
        details=details,
        error_code=error_code,
        http_status=http_status,
        job_runner_version=job_runner_version,
        progress=progress,
        started_at=started_at,
        updated_at=updated_at,
        failed_runs=failed_runs,
    )


def delete_response(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> Optional[int]:
    decrease_metric_for_artifact(kind=kind, dataset=dataset, config=config, split=split)
    return CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split).delete()


def delete_dataset_responses(dataset: str) -> int:
    """
    Delete all responses for a dataset.

    Args:
        dataset (`str`): the dataset name

    Returns:
        `int`: The number of deleted documents.
    """
    existing_cache = CachedResponseDocument.objects(dataset=dataset)
    for cache in existing_cache.no_cache():
        decrease_metric(kind=cache.kind, http_status=cache.http_status, error_code=cache.error_code)
    num_deleted_cache_responses = existing_cache.delete()
    return 0 if num_deleted_cache_responses is None else num_deleted_cache_responses


T = TypeVar("T")


@overload
def _clean_nested_mongo_object(obj: dict[str, T]) -> dict[str, T]: ...


@overload
def _clean_nested_mongo_object(obj: list[T]) -> list[T]: ...


@overload
def _clean_nested_mongo_object(obj: T) -> T: ...


def _clean_nested_mongo_object(obj: Any) -> Any:
    """get rid of BaseDict and BaseList objects from mongo (Feature.from_dict doesn't support them)"""
    if isinstance(obj, dict):
        return {k: _clean_nested_mongo_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nested_mongo_object(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_nested_mongo_object(v) for v in obj)
    else:
        return obj


class CacheEntryWithoutContent(TypedDict):
    http_status: HTTPStatus
    dataset_git_revision: str
    error_code: Optional[str]
    progress: Optional[float]
    job_runner_version: Optional[int]


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response_without_content(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithoutContent:
    try:
        response = (
            CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only("http_status", "error_code", "job_runner_version", "dataset_git_revision", "progress")
            .get()
        )
    except DoesNotExist as e:
        raise CachedArtifactNotFoundError(kind=kind, dataset=dataset, config=config, split=split) from e
    return {
        "http_status": response.http_status,
        "error_code": response.error_code,
        "dataset_git_revision": response.dataset_git_revision,
        "job_runner_version": response.job_runner_version,
        "progress": response.progress,
    }


class CacheEntryMetadata(CacheEntryWithoutContent):
    updated_at: datetime
    failed_runs: int


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response_metadata(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryMetadata:
    try:
        response = (
            CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only(
                "http_status",
                "error_code",
                "job_runner_version",
                "dataset_git_revision",
                "progress",
                "updated_at",
                "failed_runs",
            )
            .get()
        )
    except DoesNotExist as e:
        raise CachedArtifactNotFoundError(kind=kind, dataset=dataset, config=config, split=split) from e
    return {
        "http_status": response.http_status,
        "error_code": response.error_code,
        "dataset_git_revision": response.dataset_git_revision,
        "job_runner_version": response.job_runner_version,
        "progress": response.progress,
        "updated_at": response.updated_at,
        "failed_runs": response.failed_runs,
    }


class CacheEntry(CacheEntryWithoutContent):
    content: Mapping[str, Any]


class CacheEntryWithDetails(CacheEntry):
    details: Mapping[str, str]


class CachedArtifactError(Exception):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]
    cache_entry_with_details: CacheEntryWithDetails
    enhanced_details: dict[str, Any]

    def __init__(
        self,
        message: str,
        kind: str,
        dataset: str,
        config: Optional[str],
        split: Optional[str],
        cache_entry_with_details: CacheEntryWithDetails,
    ):
        super().__init__(message)
        self.kind = kind
        self.dataset = dataset
        self.config = config
        self.split = split
        self.cache_entry_with_details = cache_entry_with_details
        self.enhanced_details: dict[str, Any] = dict(self.cache_entry_with_details["details"].items())
        self.enhanced_details["copied_from_artifact"] = {
            "kind": self.kind,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
        }


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response(kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> CacheEntry:
    try:
        response = (
            CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only("content", "http_status", "error_code", "job_runner_version", "dataset_git_revision", "progress")
            .get()
        )
    except DoesNotExist as e:
        raise CachedArtifactNotFoundError(kind=kind, dataset=dataset, config=config, split=split) from e
    return {
        "content": _clean_nested_mongo_object(response.content),
        "http_status": response.http_status,
        "error_code": response.error_code,
        "job_runner_version": response.job_runner_version,
        "dataset_git_revision": response.dataset_git_revision,
        "progress": response.progress,
    }


# Note: we let the exceptions throw: it's the responsibility of the caller to manage them
def get_response_with_details(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithDetails:
    try:
        response = (
            CachedResponseDocument.objects(kind=kind, dataset=dataset, config=config, split=split)
            .only(
                "content",
                "http_status",
                "error_code",
                "job_runner_version",
                "dataset_git_revision",
                "progress",
                "details",
            )
            .get()
        )
    except DoesNotExist as e:
        raise CachedArtifactNotFoundError(kind=kind, dataset=dataset, config=config, split=split) from e
    return {
        "content": _clean_nested_mongo_object(response.content),
        "http_status": response.http_status,
        "error_code": response.error_code,
        "job_runner_version": response.job_runner_version,
        "dataset_git_revision": response.dataset_git_revision,
        "progress": response.progress,
        "details": _clean_nested_mongo_object(response.details),
    }


CACHED_RESPONSE_NOT_FOUND = "CachedResponseNotFound"
DATASET_GIT_REVISION_NOT_FOUND = "dataset-git-revision-not-found"


def get_previous_step_or_raise(
    kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> CacheEntryWithDetails:
    """
    Get the previous step from the cache, or raise an exception if it failed.

    Args:
        kind (`str`): The kind of the cache entry.
        dataset (`str`): The dataset name.
        config (`str`, *optional*): The config name.
        split (`str`, *optional*): The split name.

    Raises:
        [~`CachedArtifactNotFoundError`]: If the response does not exist.
        [~`CachedArtifactError`]: If the response is not successful.

    Returns:
        `CacheEntryWithDetails`: The response. It can be an error,
          including a cache miss (error code: `CachedResponseNotFound`)
    """
    response = get_response_with_details(kind=kind, dataset=dataset, config=config, split=split)
    if response["http_status"] != HTTPStatus.OK:
        raise CachedArtifactError(
            message="The previous step failed.",
            kind=kind,
            dataset=dataset,
            config=config,
            split=split,
            cache_entry_with_details=response,
        )
    return response


def get_all_datasets() -> set[str]:
    return set(CachedResponseDocument.objects().distinct("dataset"))


def get_datasets_with_retryable_errors() -> set[str]:
    return set(CachedResponseDocument.objects(error_code__in=ERROR_CODES_TO_RETRY).distinct("dataset"))


def is_successful_response(kind: str, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> bool:
    """
    Check if the response is successful.

    Args:
        kind (`str`): The kind of the cache entry.
        dataset (`str`): The dataset name.
        config (`str`, *optional*): The config name.
        split (`str`, *optional*): The split name.

    Raises:
        [~`mongoengine.DoesNotExist`]: If the response does not exist.

    Returns:
        `bool`: True if the response is successful, False otherwise.
    """
    return (
        CachedResponseDocument.objects(
            dataset=dataset, config=config, split=split, kind=kind, http_status=HTTPStatus.OK
        ).count()
        > 0
    )


# admin /metrics endpoint


EntriesTotalByKindStatusAndErrorCode = Mapping[tuple[str, int, Optional[str]], int]


def get_responses_count_by_kind_status_and_error_code() -> EntriesTotalByKindStatusAndErrorCode:
    """Count the number of cache entries by kind, http status and error code.

    Returns:
        an object with the total of jobs by kind, http_status and error_code.
        Keys are a tuple (kind, http_status, error_code), and values are the total of jobs.
    """
    return {
        (metric["kind"], metric["http_status"], metric["error_code"]): metric["total"]
        for metric in CachedResponseDocument.objects().aggregate(
            [
                {"$sort": {"kind": 1, "http_status": 1, "error_code": 1}},
                {
                    "$group": {
                        "_id": {"kind": "$kind", "http_status": "$http_status", "error_code": "$error_code"},
                        "total": {"$sum": 1},
                    }
                },
                {
                    "$project": {
                        "kind": "$_id.kind",
                        "http_status": "$_id.http_status",
                        "error_code": "$_id.error_code",
                        "total": "$total",
                    }
                },
            ]
        )
    }


# /cache-reports/... endpoints


class CacheReport(TypedDict):
    kind: str
    dataset: str
    dataset_git_revision: str
    config: Optional[str]
    split: Optional[str]
    http_status: int
    error_code: Optional[str]
    details: Mapping[str, Any]
    updated_at: datetime
    job_runner_version: Optional[int]
    progress: Optional[float]
    failed_runs: int


class CacheReportsPage(TypedDict):
    cache_reports: list[CacheReport]
    next_cursor: str


class InvalidCursor(Exception):
    pass


class InvalidLimit(Exception):
    pass


def get_cache_reports(kind: str, cursor: Optional[str], limit: int) -> CacheReportsPage:
    """
    Get a list of reports of the cache entries, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.

    The "reports" are the cached entries, without the "content", "details" and "updated_at" fields.

    Args:
        kind (`str`): the kind of the cache entries
        cursor (`str`):
            An opaque string value representing a pointer to a specific CachedResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.

    Raises:
        [~`simple_cache.InvalidCursor`]:
          If the cursor is invalid.
        [~`simple_cache.InvalidLimit`]:
          If the limit is an invalid number.

    Returns:
        `CacheReportsPage`: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    """
    if not cursor:
        queryset = CachedResponseDocument.objects(kind=kind)
    else:
        try:
            queryset = CachedResponseDocument.objects(kind=kind, id__gt=ObjectId(cursor))
        except InvalidId as err:
            raise InvalidCursor("Invalid cursor.") from err
    if limit <= 0:
        raise InvalidLimit("Invalid limit.")
    objects = list(queryset.order_by("+id").exclude("content").limit(limit))
    return {
        "cache_reports": [
            {
                "kind": kind,
                "dataset": object.dataset,
                "config": object.config,
                "split": object.split,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "details": _clean_nested_mongo_object(object.details),
                "updated_at": object.updated_at,
                "job_runner_version": object.job_runner_version,
                "dataset_git_revision": object.dataset_git_revision,
                "progress": object.progress,
                "failed_runs": object.failed_runs,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


def get_outdated_split_full_names_for_step(kind: str, current_version: int) -> list[SplitFullName]:
    responses = CachedResponseDocument.objects(kind=kind, job_runner_version__lt=current_version).only(
        "dataset", "config", "split"
    )
    return [
        SplitFullName(dataset=response.dataset, config=response.config, split=response.split) for response in responses
    ]


def get_dataset_responses_without_content_for_kind(kind: str, dataset: str) -> list[CacheReport]:
    responses = CachedResponseDocument.objects(kind=kind, dataset=dataset).exclude("content")
    return [
        {
            "kind": response.kind,
            "dataset": response.dataset,
            "config": response.config,
            "split": response.split,
            "http_status": response.http_status,
            "error_code": response.error_code,
            "details": _clean_nested_mongo_object(response.details),
            "updated_at": response.updated_at,
            "job_runner_version": response.job_runner_version,
            "dataset_git_revision": response.dataset_git_revision,
            "progress": response.progress,
            "failed_runs": response.failed_runs,
        }
        for response in responses
    ]


class CacheReportWithContent(CacheReport):
    content: Mapping[str, Any]


class CacheReportsWithContentPage(TypedDict):
    cache_reports_with_content: list[CacheReportWithContent]
    next_cursor: str


def get_cache_reports_with_content(kind: str, cursor: Optional[str], limit: int) -> CacheReportsWithContentPage:
    """
    Get a list of the cache report with content, along with the next cursor.
    See https://solovyov.net/blog/2020/api-pagination-design/.

    The cache reports contain all the fields of the object, including the "content" field.

    Args:
        kind (`str`): the kind of the cache entries
        cursor (`str`):
            An opaque string value representing a pointer to a specific CachedResponse item in the dataset. The
            server returns results after the given pointer.
            An empty string means to start from the beginning.
        limit (strictly positive `int`):
            The maximum number of results.

    Raises:
        [~`simple_cache.InvalidCursor`]:
          If the cursor is invalid.
        [~`simple_cache.InvalidLimit`]:
          If the limit is an invalid number.

    Returns:
        `CacheReportsWithContentPage`: A dict with the list of reports and the next cursor. The next cursor is
        an empty string if there are no more items to be fetched.
    """
    if not cursor:
        queryset = CachedResponseDocument.objects(kind=kind)
    else:
        try:
            queryset = CachedResponseDocument.objects(kind=kind, id__gt=ObjectId(cursor))
        except InvalidId as err:
            raise InvalidCursor("Invalid cursor.") from err
    if limit <= 0:
        raise InvalidLimit("Invalid limit.")
    objects = list(queryset.order_by("+id").limit(limit))
    return {
        "cache_reports_with_content": [
            {
                "kind": kind,
                "dataset": object.dataset,
                "config": object.config,
                "split": object.split,
                "http_status": object.http_status.value,
                "error_code": object.error_code,
                "content": _clean_nested_mongo_object(object.content),
                "job_runner_version": object.job_runner_version,
                "dataset_git_revision": object.dataset_git_revision,
                "details": _clean_nested_mongo_object(object.details),
                "updated_at": object.updated_at,
                "progress": object.progress,
                "failed_runs": object.failed_runs,
            }
            for object in objects
        ],
        "next_cursor": "" if len(objects) < limit else str(objects[-1].id),
    }


class CacheEntryFullMetadata(CacheEntryMetadata):
    kind: str
    dataset: str
    config: Optional[str]
    split: Optional[str]


def _get_df(entries: list[CacheEntryFullMetadata]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": pd.Series([entry["kind"] for entry in entries], dtype="category"),
            "dataset": pd.Series([entry["dataset"] for entry in entries], dtype="str"),
            "config": pd.Series([entry["config"] for entry in entries], dtype="str"),
            "split": pd.Series([entry["split"] for entry in entries], dtype="str"),
            "http_status": pd.Series(
                [entry["http_status"] for entry in entries], dtype="category"
            ),  # check if it's working as expected
            "error_code": pd.Series([entry["error_code"] for entry in entries], dtype="category"),
            "dataset_git_revision": pd.Series([entry["dataset_git_revision"] for entry in entries], dtype="str"),
            "job_runner_version": pd.Series([entry["job_runner_version"] for entry in entries], dtype=pd.Int16Dtype()),
            "progress": pd.Series([entry["progress"] for entry in entries], dtype="float"),
            "updated_at": pd.Series(
                [entry["updated_at"] for entry in entries], dtype="datetime64[ns]"
            ),  # check if it's working as expected
            "failed_runs": pd.Series([entry["failed_runs"] for entry in entries], dtype=pd.Int16Dtype()),
        }
    )
    # ^ does not seem optimal at all, but I get the types right


def get_cache_entries_df(dataset: str, cache_kinds: Optional[list[str]] = None) -> pd.DataFrame:
    filters = {"dataset": dataset}
    if cache_kinds:
        filters["kind"] = {"$in": cache_kinds}  # type: ignore
    return CachedResponseDocument.fetch_as_df(query=filters)


def get_cache_entries_df_old(dataset: str, cache_kinds: Optional[list[str]] = None) -> pd.DataFrame:
    filters = {}
    if cache_kinds:
        filters["kind__in"] = cache_kinds
    return _get_df(
        [
            {
                "kind": response.kind,
                "dataset": response.dataset,
                "config": response.config,
                "split": response.split,
                "http_status": response.http_status,
                "error_code": response.error_code,
                "dataset_git_revision": response.dataset_git_revision,
                "job_runner_version": response.job_runner_version,
                "progress": response.progress,
                "updated_at": response.updated_at,
                "failed_runs": response.failed_runs,
            }
            for response in CachedResponseDocument.objects(dataset=dataset, **filters).only(
                "kind",
                "dataset",
                "config",
                "split",
                "http_status",
                "error_code",
                "job_runner_version",
                "dataset_git_revision",
                "progress",
                "updated_at",
                "failed_runs",
            )
        ]
    )


def get_cache_count_for_dataset(dataset: str) -> int:
    return CachedResponseDocument.objects(dataset=dataset).count()


def has_some_cache(dataset: str) -> bool:
    return get_cache_count_for_dataset(dataset) > 0


def fetch_names(dataset: str, config: Optional[str], cache_kind: str, names_field: str, name_field: str) -> list[str]:
    """
    Fetch a list of names from the cache database.

    If no entry is found in cache, return an empty list. Exceptions are silently caught.

    Args:
        dataset (`str`): The dataset name.
        config (`str`, *optional*): The config name. Only needed for split names.
        cache_kind (`str`): The cache kind to fetch, eg "dataset-config-names".
        names_field (`str`): The name of the field containing the list of names, eg: "config_names", or "splits".
        name_field (`str`): The name of the field containing the name, eg: "config", or "split".

    Returns:
        `list[str]`: The list of names.
    """
    try:
        names = []
        response = get_response_with_details(kind=cache_kind, dataset=dataset, config=config)
        for name_item in response["content"][names_field]:
            name = name_item[name_field]
            if not isinstance(name, str):
                raise ValueError(f"Invalid name: {name}, type should be str, got: {type(name)}")
            names.append(name)
        return names
    except Exception:
        return []


def get_datasets_with_last_updated_kind(kind: str, days: int) -> list[str]:
    """
    Get the list of datasets for which an artifact of some kind has been updated in the last days.

    Args:
        kind (`str`): The kind of the cache entries.
        days (`int`): The number of days to look back.

    Returns:
        `list[str]`: The list of datasets.
    """

    pipeline = [
        {"$match": {"kind": kind, "http_status": HTTPStatus.OK, "updated_at": {"$gt": get_datetime(days=days)}}},
        {"$sort": {"updated_at": 1}},
        {"$group": {"_id": 0, "datasets": {"$addToSet": "$dataset"}}},
        {"$unwind": "$datasets"},
    ]
    return [
        str(dataset["datasets"])
        for dataset in CachedResponseDocument.objects(
            kind=kind, http_status=HTTPStatus.OK, updated_at__gt=get_datetime(days=days)
        ).aggregate(pipeline)
    ]


# only for the tests
def _clean_cache_database() -> None:
    CachedResponseDocument.drop_collection()  # type: ignore
    CacheTotalMetricDocument.drop_collection()  # type: ignore

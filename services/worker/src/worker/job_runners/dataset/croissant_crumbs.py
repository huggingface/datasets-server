# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any

from datasets import ClassLabel, Features, Image, Value
from libcommon.constants import CROISSANT_MAX_CONFIGS
from libcommon.croissant_utils import get_record_set, feature_to_croissant_field
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
)

from worker.dtos import CompleteJobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner

NAME_PATTERN_REGEX = "[^a-zA-Z0-9\\-_\\.]"


def _escape_name(name: str, names: set[str]) -> str:
    """Escapes names and IDs in Croissant.

    Reasons:
    - `/` are used in the syntax as delimiters. So we replace them.
    - Two FileObject/FileSet/RecordSet/Fields cannot have the same ID. So we append a postfix in case it happens.

    Args:
        name: The initial non-escaped name.
        names: The set of already existing names.
    Returns:
        `str`: The escaped name.
    """
    escaped_name = re.sub(NAME_PATTERN_REGEX, "_", name)
    while escaped_name in names:
        escaped_name = f"{escaped_name}_0"
    names.add(escaped_name)
    return escaped_name


def _remove_none_values(json: Mapping[str, Any]) -> Mapping[str, Any]:
    """Removes None values in the first depth of a dict."""
    return {k: v for k, v in json.items() if v is not None}


def get_croissant_crumbs_from_dataset_infos(
    dataset: str, infos: list[Mapping[str, Any]], partial: bool, truncated_configs: bool
) -> Mapping[str, Any]:
    """Generates the "crumbs" of the Croissant JSON-LD metadata from the dataset infos.

    It's only a subset of the full JSON-LD metadata. See the Hugging Face API `/croissant` endpoint
    to get the complete Croissant JSON-LD metadata.
    """
    repo_name = "repo"
    names: set[str] = set(repo_name)
    distribution = [
        _remove_none_values(
            {
                "@type": "cr:FileObject",
                "@id": repo_name,
                "name": repo_name,
                "description": "The Hugging Face git repository.",
                "contentUrl": f"https://huggingface.co/datasets/{dataset}/tree/refs%2Fconvert%2Fparquet",
                "encodingFormat": "git+https",
                "sha256": "https://github.com/mlcommons/croissant/issues/80",
            }
        )
    ]
    record_set = []
    for info in infos:
        description_body = ""
        config = info["config_name"]
        features = Features.from_dict(info["features"])
        fields: list[dict[str, Any]] = []
        splits = list(info["splits"])
        distribution_name = _escape_name(f"parquet-files-for-config-{config}", names)
        distribution.append(
            _remove_none_values(
                {
                    "@type": "cr:FileSet",
                    "@id": distribution_name,
                    "name": distribution_name,
                    "description": "The underlying Parquet files as converted by Hugging Face (see: https://huggingface.co/docs/datasets-server/parquet).",
                    "containedIn": {"@id": repo_name},
                    "encodingFormat": "application/x-parquet",
                    "includes": f"{config}/*/*.parquet",
                }
            )
        )
        skipped_columns = []
        record_set_name = get_record_set(dataset=dataset, config_name=config)
        record_set_name = _escape_name(record_set_name, names)
        for column, feature in features.items():
            fields_names: set[str] = set()
            field_name = f"{record_set_name}/{_escape_name(column, fields_names)}"
            field = feature_to_croissant_field(distribution_name, field_name, column, feature)
            if field:
                fields.append(field)
            else:
                skipped_columns.append(column)
        description = f"{dataset} - '{config}' subset"
        if partial:
            description += " (first 5GB)"
        if truncated_configs:
            description += f" (only the first {CROISSANT_MAX_CONFIGS} subsets are included in this metadata)"
        if len(splits) > 1:
            description_body += f"\n- {len(splits)} split{'s' if len(splits) > 1 else ''}: {', '.join(splits)}"
        if skipped_columns:
            description_body += f"\n- {len(skipped_columns)} skipped column{'s' if len(skipped_columns) > 1 else ''}: {', '.join(skipped_columns)}"
        if description_body:
            description += "\n\nAdditional information:"
            description += description_body
        record_set.append(
            _remove_none_values(
                {
                    "@type": "cr:RecordSet",
                    "@id": record_set_name,
                    "name": record_set_name,
                    "description": description,
                    "field": fields,
                }
            )
        )
    context = {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataBiases": "cr:dataBiases",
        "dataCollection": "cr:dataCollection",
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "personalSensitiveInformation": "cr:personalSensitiveInformation",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }
    return _remove_none_values(
        {
            "@context": context,
            "@type": "sc:Dataset",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "distribution": distribution,
            "recordSet": record_set,
        }
    )


def compute_croissant_crumbs_response(dataset: str) -> Mapping[str, Any]:
    """
    Get the response of 'dataset-croissant-crumbs' for one specific dataset on huggingface.co.

    If the dataset contains more than 100 configs, only the first 100 configs are included in the croissant metadata parts.

    Here, we don't truncate the number of fields. See the /croissant-crumbs endpoint implementation in services/api for
      truncation of the fields.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns `dict[str, Any]`: the croissant metadata parts.
    """
    logging.info(f"compute 'dataset-croissant-crumbs' for {dataset=}")

    dataset_info_response = get_previous_step_or_raise(kind="dataset-info", dataset=dataset)
    try:
        content = dataset_info_response["content"]
        truncated_configs = len(content["dataset_info"]) > CROISSANT_MAX_CONFIGS
        infos = list(islice(content["dataset_info"].values(), CROISSANT_MAX_CONFIGS))
        partial = content["partial"]
        croissant_crumbs = get_croissant_crumbs_from_dataset_infos(
            dataset=dataset, infos=infos, partial=partial, truncated_configs=truncated_configs
        )
    except KeyError as e:
        raise PreviousStepFormatError("Previous step 'dataset-info' did not return the expected content.", e) from e
    return croissant_crumbs


class DatasetCroissantCrumbsJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-croissant-crumbs"

    def compute(self) -> CompleteJobResult:
        response_content = compute_croissant_crumbs_response(dataset=self.dataset)
        return CompleteJobResult(response_content)

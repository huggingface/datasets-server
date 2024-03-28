# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from worker.job_runners.dataset.croissant_crumbs import (
    get_croissant_crumbs_from_dataset_infos,
)

squad_info = {
    "description": "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.\n",
    "citation": '@article{2016arXiv160605250R,\n       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},\n                 Konstantin and {Liang}, Percy},\n        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",\n      journal = {arXiv e-prints},\n         year = 2016,\n          eid = {arXiv:1606.05250},\n        pages = {arXiv:1606.05250},\narchivePrefix = {arXiv},\n       eprint = {1606.05250},\n}\n',
    "homepage": "https://rajpurkar.github.io/SQuAD-explorer/",
    "license": ["mit"],
    "tags": ["foo", "doi:hf/123456789", "region:us"],
    "features": {
        "id": {"dtype": "string", "_type": "Value"},
        "title": {"dtype": "string", "_type": "Value"},
        "context": {"dtype": "string", "_type": "Value"},
        "question": {"dtype": "string", "_type": "Value"},
        "answers": {
            "feature": {
                "text": {"dtype": "string", "_type": "Value"},
                "answer_start": {"dtype": "int32", "_type": "Value"},
            },
            "_type": "Sequence",
        },
    },
    "task_templates": [{"task": "question-answering-extractive"}],
    "builder_name": "squad",
    "config_name": "user/squad with space",
    "version": {"version_str": "1.0.0", "description": "", "major": 1, "minor": 0, "patch": 0},
    "splits": {
        "train": {"name": "train", "num_bytes": 79346108, "num_examples": 87599, "dataset_name": "squad"},
        "validation": {
            "name": "validation",
            "num_bytes": 10472984,
            "num_examples": 10570,
            "dataset_name": "squad",
        },
    },
    "download_checksums": {
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json": {
            "num_bytes": 30288272,
            "checksum": None,
        },
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json": {
            "num_bytes": 4854279,
            "checksum": None,
        },
    },
    "download_size": 35142551,
    "dataset_size": 89819092,
    "size_in_bytes": 124961643,
}


v1_context = {
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


def test_get_croissant_context_from_dataset_infos() -> None:
    croissant_crumbs = get_croissant_crumbs_from_dataset_infos(
        "user/squad with space", [squad_info, squad_info], partial=False, truncated_configs=False
    )
    assert croissant_crumbs["@context"] == v1_context


def test_get_croissant_crumbs_from_dataset_infos() -> None:
    croissant_crumbs = get_croissant_crumbs_from_dataset_infos(
        "user/squad with space", [squad_info, squad_info], partial=False, truncated_configs=False
    )
    assert "@context" in croissant_crumbs
    assert "@type" in croissant_crumbs

    # Test recordSet.
    assert "recordSet" in croissant_crumbs
    assert croissant_crumbs["recordSet"]
    assert isinstance(croissant_crumbs["recordSet"], list)
    assert len(croissant_crumbs["recordSet"]) == 2
    assert croissant_crumbs["recordSet"][0]["@type"] == croissant_crumbs["recordSet"][1]["@type"] == "cr:RecordSet"
    assert croissant_crumbs["recordSet"][0]["name"] == "record_set_user_squad_with_space"
    assert croissant_crumbs["recordSet"][1]["name"] == "record_set_user_squad_with_space_0"
    assert isinstance(croissant_crumbs["recordSet"][0]["field"], list)
    assert isinstance(squad_info["features"], dict)
    assert "1 skipped column: answers" in croissant_crumbs["recordSet"][0]["description"]
    assert croissant_crumbs["recordSet"][0]["@id"] == "record_set_user_squad_with_space"
    assert croissant_crumbs["recordSet"][1]["@id"] == "record_set_user_squad_with_space_0"
    for i, _ in enumerate(croissant_crumbs["recordSet"]):
        for field in croissant_crumbs["recordSet"][i]["field"]:
            assert "source" in field
            assert "fileSet" in field["source"]
            assert "@id" in field["source"]["fileSet"]
            assert field["source"]["fileSet"]["@id"]
            assert "extract" in field["source"]
            assert field["source"]["extract"]["column"] == field["@id"].split("/")[-1]

    # Test fields.
    assert len(croissant_crumbs["recordSet"][0]["field"]) == 4
    assert len(croissant_crumbs["recordSet"][1]["field"]) == 4
    for field in croissant_crumbs["recordSet"][0]["field"]:
        assert field["@type"] == "cr:Field"
        assert field["dataType"] == "sc:Text"
    assert len(croissant_crumbs["recordSet"][0]["field"]) == len(squad_info["features"]) - 1

    # Test distribution.
    assert "distribution" in croissant_crumbs
    assert croissant_crumbs["distribution"]
    assert isinstance(croissant_crumbs["distribution"], list)
    assert croissant_crumbs["distribution"][0]["@type"] == "cr:FileObject"
    assert croissant_crumbs["distribution"][1]["@type"] == "cr:FileSet"
    assert croissant_crumbs["distribution"][2]["@type"] == "cr:FileSet"
    assert croissant_crumbs["distribution"][0]["name"] == "repo"
    for distribution in croissant_crumbs["distribution"]:
        assert "@id" in distribution
        if "containedIn" in distribution:
            assert "@id" in distribution["containedIn"]

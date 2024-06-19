# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable, Mapping
from dataclasses import replace
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runners.split.presidio_scan import (
    SplitPresidioEntitiesScanJobRunner,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest, get_default_config_split
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, str, AppConfig], SplitPresidioEntitiesScanJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitPresidioEntitiesScanJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitPresidioEntitiesScanJobRunner(
            job_info={
                "type": SplitPresidioEntitiesScanJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 70,
                "started_at": None,
            },
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


DEFAULT_EMPTY_RESPONSE = {
    "scanned_columns": [],
    "num_in_vehicle_registration_entities": 0,
    "num_organization_entities": 0,
    "num_sg_nric_fin_entities": 0,
    "num_person_entities": 0,
    "num_credit_card_entities": 0,
    "num_medical_license_entities": 0,
    "num_nrp_entities": 0,
    "num_us_ssn_entities": 0,
    "num_crypto_entities": 0,
    "num_date_time_entities": 0,
    "num_location_entities": 0,
    "num_us_driver_license_entities": 0,
    "num_phone_number_entities": 0,
    "num_url_entities": 0,
    "num_us_passport_entities": 0,
    "num_age_entities": 0,
    "num_au_acn_entities": 0,
    "num_email_address_entities": 0,
    "num_in_pan_entities": 0,
    "num_ip_address_entities": 0,
    "num_id_entities": 0,
    "num_us_bank_number_entities": 0,
    "num_in_aadhaar_entities": 0,
    "num_us_itin_entities": 0,
    "num_au_medicare_entities": 0,
    "num_iban_code_entities": 0,
    "num_au_tfn_entities": 0,
    "num_uk_nhs_entities": 0,
    "num_email_entities": 0,
    "num_au_abn_entities": 0,
    "num_rows_with_in_vehicle_registration_entities": 0,
    "num_rows_with_organization_entities": 0,
    "num_rows_with_sg_nric_fin_entities": 0,
    "num_rows_with_person_entities": 0,
    "num_rows_with_credit_card_entities": 0,
    "num_rows_with_medical_license_entities": 0,
    "num_rows_with_nrp_entities": 0,
    "num_rows_with_us_ssn_entities": 0,
    "num_rows_with_crypto_entities": 0,
    "num_rows_with_date_time_entities": 0,
    "num_rows_with_location_entities": 0,
    "num_rows_with_us_driver_license_entities": 0,
    "num_rows_with_phone_number_entities": 0,
    "num_rows_with_url_entities": 0,
    "num_rows_with_us_passport_entities": 0,
    "num_rows_with_age_entities": 0,
    "num_rows_with_au_acn_entities": 0,
    "num_rows_with_email_address_entities": 0,
    "num_rows_with_in_pan_entities": 0,
    "num_rows_with_ip_address_entities": 0,
    "num_rows_with_id_entities": 0,
    "num_rows_with_us_bank_number_entities": 0,
    "num_rows_with_in_aadhaar_entities": 0,
    "num_rows_with_us_itin_entities": 0,
    "num_rows_with_au_medicare_entities": 0,
    "num_rows_with_iban_code_entities": 0,
    "num_rows_with_au_tfn_entities": 0,
    "num_rows_with_uk_nhs_entities": 0,
    "num_rows_with_email_entities": 0,
    "num_rows_with_au_abn_entities": 0,
    "num_scanned_rows": 0,
    "has_scanned_columns": False,
    "full_scan": None,
    "entities": [],
}


@pytest.mark.parametrize(
    "name,rows_max_number,expected_content",
    [
        (
            "public",
            100_000,
            DEFAULT_EMPTY_RESPONSE,
        ),
        (
            "presidio_scan",
            100_000,  # dataset has less rows
            {
                "scanned_columns": ["col"],
                "num_in_vehicle_registration_entities": 0,
                "num_organization_entities": 0,
                "num_sg_nric_fin_entities": 0,
                "num_person_entities": 2,
                "num_credit_card_entities": 0,
                "num_medical_license_entities": 0,
                "num_nrp_entities": 0,
                "num_us_ssn_entities": 1,
                "num_crypto_entities": 0,
                "num_date_time_entities": 0,
                "num_location_entities": 0,
                "num_us_driver_license_entities": 0,
                "num_phone_number_entities": 0,
                "num_url_entities": 0,
                "num_us_passport_entities": 0,
                "num_age_entities": 0,
                "num_au_acn_entities": 0,
                "num_email_address_entities": 1,
                "num_in_pan_entities": 0,
                "num_ip_address_entities": 1,
                "num_id_entities": 0,
                "num_us_bank_number_entities": 0,
                "num_in_aadhaar_entities": 0,
                "num_us_itin_entities": 0,
                "num_au_medicare_entities": 0,
                "num_iban_code_entities": 0,
                "num_au_tfn_entities": 0,
                "num_uk_nhs_entities": 0,
                "num_email_entities": 0,
                "num_au_abn_entities": 0,
                "num_rows_with_in_vehicle_registration_entities": 0,
                "num_rows_with_organization_entities": 0,
                "num_rows_with_sg_nric_fin_entities": 0,
                "num_rows_with_person_entities": 2,
                "num_rows_with_credit_card_entities": 0,
                "num_rows_with_medical_license_entities": 0,
                "num_rows_with_nrp_entities": 0,
                "num_rows_with_us_ssn_entities": 1,
                "num_rows_with_crypto_entities": 0,
                "num_rows_with_date_time_entities": 0,
                "num_rows_with_location_entities": 0,
                "num_rows_with_us_driver_license_entities": 0,
                "num_rows_with_phone_number_entities": 0,
                "num_rows_with_url_entities": 0,
                "num_rows_with_us_passport_entities": 0,
                "num_rows_with_age_entities": 0,
                "num_rows_with_au_acn_entities": 0,
                "num_rows_with_email_address_entities": 1,
                "num_rows_with_in_pan_entities": 0,
                "num_rows_with_ip_address_entities": 1,
                "num_rows_with_id_entities": 0,
                "num_rows_with_us_bank_number_entities": 0,
                "num_rows_with_in_aadhaar_entities": 0,
                "num_rows_with_us_itin_entities": 0,
                "num_rows_with_au_medicare_entities": 0,
                "num_rows_with_iban_code_entities": 0,
                "num_rows_with_au_tfn_entities": 0,
                "num_rows_with_uk_nhs_entities": 0,
                "num_rows_with_email_entities": 0,
                "num_rows_with_au_abn_entities": 0,
                "num_scanned_rows": 6,
                "has_scanned_columns": True,
                "full_scan": True,
                "entities": [
                    {"column_name": "col", "row_idx": 0, "text": "Gi****** Gi*****", "type": "PERSON"},
                    {"column_name": "col", "row_idx": 1, "text": "Gi*****", "type": "PERSON"},
                    {"column_name": "col", "row_idx": 2, "text": "19*.***.*.*", "type": "IP_ADDRESS"},
                    {"column_name": "col", "row_idx": 3, "text": "34*-**-****", "type": "US_SSN"},
                    {
                        "column_name": "col",
                        "row_idx": 4,
                        "text": "gi******.*******@********.***",
                        "type": "EMAIL_ADDRESS",
                    },
                ],
            },
        ),
        (
            "presidio_scan",
            3,  # dataset has more rows
            {
                "scanned_columns": ["col"],
                "num_in_vehicle_registration_entities": 0,
                "num_organization_entities": 0,
                "num_sg_nric_fin_entities": 0,
                "num_person_entities": 2,
                "num_credit_card_entities": 0,
                "num_medical_license_entities": 0,
                "num_nrp_entities": 0,
                "num_us_ssn_entities": 0,
                "num_crypto_entities": 0,
                "num_date_time_entities": 0,
                "num_location_entities": 0,
                "num_us_driver_license_entities": 0,
                "num_phone_number_entities": 0,
                "num_url_entities": 0,
                "num_us_passport_entities": 0,
                "num_age_entities": 0,
                "num_au_acn_entities": 0,
                "num_email_address_entities": 0,
                "num_in_pan_entities": 0,
                "num_ip_address_entities": 1,
                "num_id_entities": 0,
                "num_us_bank_number_entities": 0,
                "num_in_aadhaar_entities": 0,
                "num_us_itin_entities": 0,
                "num_au_medicare_entities": 0,
                "num_iban_code_entities": 0,
                "num_au_tfn_entities": 0,
                "num_uk_nhs_entities": 0,
                "num_email_entities": 0,
                "num_au_abn_entities": 0,
                "num_rows_with_in_vehicle_registration_entities": 0,
                "num_rows_with_organization_entities": 0,
                "num_rows_with_sg_nric_fin_entities": 0,
                "num_rows_with_person_entities": 2,
                "num_rows_with_credit_card_entities": 0,
                "num_rows_with_medical_license_entities": 0,
                "num_rows_with_nrp_entities": 0,
                "num_rows_with_us_ssn_entities": 0,
                "num_rows_with_crypto_entities": 0,
                "num_rows_with_date_time_entities": 0,
                "num_rows_with_location_entities": 0,
                "num_rows_with_us_driver_license_entities": 0,
                "num_rows_with_phone_number_entities": 0,
                "num_rows_with_url_entities": 0,
                "num_rows_with_us_passport_entities": 0,
                "num_rows_with_age_entities": 0,
                "num_rows_with_au_acn_entities": 0,
                "num_rows_with_email_address_entities": 0,
                "num_rows_with_in_pan_entities": 0,
                "num_rows_with_ip_address_entities": 1,
                "num_rows_with_id_entities": 0,
                "num_rows_with_us_bank_number_entities": 0,
                "num_rows_with_in_aadhaar_entities": 0,
                "num_rows_with_us_itin_entities": 0,
                "num_rows_with_au_medicare_entities": 0,
                "num_rows_with_iban_code_entities": 0,
                "num_rows_with_au_tfn_entities": 0,
                "num_rows_with_uk_nhs_entities": 0,
                "num_rows_with_email_entities": 0,
                "num_rows_with_au_abn_entities": 0,
                "num_scanned_rows": 3,
                "has_scanned_columns": True,
                "full_scan": False,
                "entities": [
                    {"column_name": "col", "row_idx": 0, "text": "Gi****** Gi*****", "type": "PERSON"},
                    {"column_name": "col", "row_idx": 1, "text": "Gi*****", "type": "PERSON"},
                    {"column_name": "col", "row_idx": 2, "text": "19*.***.*.*", "type": "IP_ADDRESS"},
                ],
            },
        ),
        (
            "presidio_scan",
            6,  # dataset has same amount of rows
            {
                "scanned_columns": ["col"],
                "num_in_vehicle_registration_entities": 0,
                "num_organization_entities": 0,
                "num_sg_nric_fin_entities": 0,
                "num_person_entities": 2,
                "num_credit_card_entities": 0,
                "num_medical_license_entities": 0,
                "num_nrp_entities": 0,
                "num_us_ssn_entities": 1,
                "num_crypto_entities": 0,
                "num_date_time_entities": 0,
                "num_location_entities": 0,
                "num_us_driver_license_entities": 0,
                "num_phone_number_entities": 0,
                "num_url_entities": 0,
                "num_us_passport_entities": 0,
                "num_age_entities": 0,
                "num_au_acn_entities": 0,
                "num_email_address_entities": 1,
                "num_in_pan_entities": 0,
                "num_ip_address_entities": 1,
                "num_id_entities": 0,
                "num_us_bank_number_entities": 0,
                "num_in_aadhaar_entities": 0,
                "num_us_itin_entities": 0,
                "num_au_medicare_entities": 0,
                "num_iban_code_entities": 0,
                "num_au_tfn_entities": 0,
                "num_uk_nhs_entities": 0,
                "num_email_entities": 0,
                "num_au_abn_entities": 0,
                "num_rows_with_in_vehicle_registration_entities": 0,
                "num_rows_with_organization_entities": 0,
                "num_rows_with_sg_nric_fin_entities": 0,
                "num_rows_with_person_entities": 2,
                "num_rows_with_credit_card_entities": 0,
                "num_rows_with_medical_license_entities": 0,
                "num_rows_with_nrp_entities": 0,
                "num_rows_with_us_ssn_entities": 1,
                "num_rows_with_crypto_entities": 0,
                "num_rows_with_date_time_entities": 0,
                "num_rows_with_location_entities": 0,
                "num_rows_with_us_driver_license_entities": 0,
                "num_rows_with_phone_number_entities": 0,
                "num_rows_with_url_entities": 0,
                "num_rows_with_us_passport_entities": 0,
                "num_rows_with_age_entities": 0,
                "num_rows_with_au_acn_entities": 0,
                "num_rows_with_email_address_entities": 1,
                "num_rows_with_in_pan_entities": 0,
                "num_rows_with_ip_address_entities": 1,
                "num_rows_with_id_entities": 0,
                "num_rows_with_us_bank_number_entities": 0,
                "num_rows_with_in_aadhaar_entities": 0,
                "num_rows_with_us_itin_entities": 0,
                "num_rows_with_au_medicare_entities": 0,
                "num_rows_with_iban_code_entities": 0,
                "num_rows_with_au_tfn_entities": 0,
                "num_rows_with_uk_nhs_entities": 0,
                "num_rows_with_email_entities": 0,
                "num_rows_with_au_abn_entities": 0,
                "num_scanned_rows": 6,
                "has_scanned_columns": True,
                "full_scan": True,
                "entities": [
                    {"column_name": "col", "row_idx": 0, "text": "Gi****** Gi*****", "type": "PERSON"},
                    {"column_name": "col", "row_idx": 1, "text": "Gi*****", "type": "PERSON"},
                    {"column_name": "col", "row_idx": 2, "text": "19*.***.*.*", "type": "IP_ADDRESS"},
                    {"column_name": "col", "row_idx": 3, "text": "34*-**-****", "type": "US_SSN"},
                    {
                        "column_name": "col",
                        "row_idx": 4,
                        "text": "gi******.*******@********.***",
                        "type": "EMAIL_ADDRESS",
                    },
                ],
            },
        ),
    ],
)
def test_compute(
    hub_responses_public: HubDatasetTest,
    hub_responses_presidio_scan: HubDatasetTest,
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    name: str,
    rows_max_number: int,
    expected_content: Mapping[str, Any],
) -> None:
    hub_datasets = {"public": hub_responses_public, "presidio_scan": hub_responses_presidio_scan}
    dataset = hub_datasets[name]["name"]
    upstream_content = hub_datasets[name]["parquet_and_info_response"]
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, presidio_scan=replace(app_config.presidio_scan, rows_max_number=rows_max_number)),
    )
    upsert_response(
        kind="config-parquet-and-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content=upstream_content,
        job_runner_version=1,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    response = job_runner.compute()
    assert response
    assert response.content == expected_content


@pytest.mark.parametrize(
    "dataset,columns_max_number,upstream_content,upstream_status,exception_name",
    [
        ("DVUser/doesnotexist", 10, {}, HTTPStatus.OK, "CachedArtifactNotFoundError"),
        ("DVUser/wrong_format", 10, {}, HTTPStatus.OK, "PreviousStepFormatError"),
        (
            "DVUser/upstream_failed",
            10,
            {},
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "CachedArtifactError",
        ),
    ],
)
def test_compute_failed(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    columns_max_number: int,
    upstream_content: Mapping[str, Any],
    upstream_status: HTTPStatus,
    exception_name: str,
) -> None:
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, urls_scan=replace(app_config.urls_scan, columns_max_number=columns_max_number)),
    )
    if dataset != "DVUser/doesnotexist":
        upsert_response(
            kind="config-parquet-and-info",
            dataset=dataset,
            config=config,
            content=upstream_content,
            dataset_git_revision=REVISION_NAME,
            job_runner_version=1,
            progress=1.0,
            http_status=upstream_status,
        )
    with pytest.raises(Exception) as exc_info:
        job_runner.compute()
    assert exc_info.typename == exception_name

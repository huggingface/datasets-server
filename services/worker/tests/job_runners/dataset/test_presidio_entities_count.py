# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactNotFoundError, upsert_response

from worker.config import AppConfig
from worker.dtos import PresidioEntitiesScanResponse
from worker.job_runners.dataset.presidio_entities_count import (
    DatasetPresidioEntitiesCountJobRunner,
)

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetPresidioEntitiesCountJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetPresidioEntitiesCountJobRunner:
        return DatasetPresidioEntitiesCountJobRunner(
            job_info={
                "type": DatasetPresidioEntitiesCountJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": None,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        )

    return _get_job_runner


SAMPLE_RESPONSE = PresidioEntitiesScanResponse(
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
    }
)

SAMPLE_RESPONSE_NOT_FULL_SCAN = PresidioEntitiesScanResponse(
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
    }
)


@pytest.mark.parametrize(
    "dataset,split_names_status,split_names_content,split_upstream_status"
    + ",split_upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok_full_scan",
            HTTPStatus.OK,
            {
                "splits": [
                    {"dataset": "dataset_ok_full_scan", "config": "config1", "split": "split1"},
                    {"dataset": "dataset_ok_full_scan", "config": "config1", "split": "split2"},
                    {"dataset": "dataset_ok_full_scan", "config": "config2", "split": "split3"},
                ]
            },
            [HTTPStatus.OK] * 3,
            [SAMPLE_RESPONSE] * 3,
            None,
            {
                "scanned_columns": SAMPLE_RESPONSE["scanned_columns"],
                "num_rows_with_person_entities": SAMPLE_RESPONSE["num_rows_with_person_entities"] * 3,
                "num_rows_with_phone_number_entities": SAMPLE_RESPONSE["num_rows_with_phone_number_entities"] * 3,
                "num_rows_with_email_address_entities": SAMPLE_RESPONSE["num_rows_with_email_address_entities"] * 3,
                "num_rows_with_sensitive_pii": (
                    SAMPLE_RESPONSE["num_rows_with_credit_card_entities"]
                    + SAMPLE_RESPONSE["num_rows_with_us_ssn_entities"]
                    + SAMPLE_RESPONSE["num_rows_with_us_passport_entities"]
                    + SAMPLE_RESPONSE["num_rows_with_iban_code_entities"]
                )
                * 3,
                "num_scanned_rows": SAMPLE_RESPONSE["num_scanned_rows"] * 3,
                "has_scanned_columns": True,
                "full_scan": True,
            },
            False,
        ),
        (
            "dataset_ok_not_full_scan",
            HTTPStatus.OK,
            {
                "splits": [
                    {"dataset": "dataset_ok_not_full_scan", "config": "config1", "split": "split1"},
                    {"dataset": "dataset_ok_not_full_scan", "config": "config1", "split": "split2"},
                    {"dataset": "dataset_ok_not_full_scan", "config": "config2", "split": "split3"},
                ]
            },
            [HTTPStatus.OK] * 3,
            [SAMPLE_RESPONSE_NOT_FULL_SCAN] * 3,
            None,
            {
                "scanned_columns": SAMPLE_RESPONSE_NOT_FULL_SCAN["scanned_columns"],
                "num_rows_with_person_entities": SAMPLE_RESPONSE_NOT_FULL_SCAN["num_rows_with_person_entities"] * 3,
                "num_rows_with_phone_number_entities": SAMPLE_RESPONSE_NOT_FULL_SCAN[
                    "num_rows_with_phone_number_entities"
                ]
                * 3,
                "num_rows_with_email_address_entities": SAMPLE_RESPONSE_NOT_FULL_SCAN[
                    "num_rows_with_email_address_entities"
                ]
                * 3,
                "num_rows_with_sensitive_pii": (
                    SAMPLE_RESPONSE_NOT_FULL_SCAN["num_rows_with_credit_card_entities"]
                    + SAMPLE_RESPONSE_NOT_FULL_SCAN["num_rows_with_us_ssn_entities"]
                    + SAMPLE_RESPONSE_NOT_FULL_SCAN["num_rows_with_us_passport_entities"]
                    + SAMPLE_RESPONSE_NOT_FULL_SCAN["num_rows_with_iban_code_entities"]
                )
                * 3,
                "num_scanned_rows": SAMPLE_RESPONSE_NOT_FULL_SCAN["num_scanned_rows"] * 3,
                "has_scanned_columns": True,
                "full_scan": False,
            },
            False,
        ),
        (
            "previous_step_error",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {},
            [],
            [],
            "CachedArtifactError",
            None,
            True,
        ),
        (
            "previous_step_format_error",
            HTTPStatus.OK,
            {
                "splits": [
                    {"dataset": "dataset_ok_full_scan", "config": "config1", "split": "split1"},
                    {"dataset": "dataset_ok_full_scan", "config": "config1", "split": "split2"},
                    {"dataset": "dataset_ok_full_scan", "config": "config2", "split": "split3"},
                ]
            },
            [HTTPStatus.OK],
            [{"wrong_format": None}],
            "PreviousStepFormatError",
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    split_names_status: HTTPStatus,
    split_names_content: Any,
    split_upstream_status: list[HTTPStatus],
    split_upstream_content: list[Any],
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="dataset-split-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content=split_names_content,
        http_status=split_names_status,
    )

    if split_names_status == HTTPStatus.OK:
        for split_item, status, content in zip(
            split_names_content["splits"], split_upstream_status, split_upstream_content
        ):
            upsert_response(
                kind="split-presidio-scan",
                dataset=dataset,
                dataset_git_revision=REVISION_NAME,
                config=split_item["config"],
                split=split_item["split"],
                content=content,
                http_status=status,
            )

    job_runner = get_job_runner(dataset, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

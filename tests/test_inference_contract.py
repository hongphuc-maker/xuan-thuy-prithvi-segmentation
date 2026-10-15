from __future__ import annotations

import copy

import pytest

from deployment.contracts import validate_inference_manifest


def manifest():
    dates = [
        "2027-01-08",
        "2027-05-25",
        "2027-07-17",
        "2027-08-11",
        "2027-09-02",
        "2027-09-20",
    ]
    return {
        "schema": "prithvi-inference-dataset-v1",
        "protocol": "A_temporal_transfer",
        "dataset_id": "xuan_thuy_2027",
        "region_id": "xuan_thuy",
        "target_date": dates[-1],
        "images": [
            {"path": f"rasters/{value}.tif", "date": value, "sha256": "a" * 64}
            for value in dates
        ],
        "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
        "radiometry": {
            "dn_offset": -1000.0,
            "scale": 10000.0,
            "offset_verified_for_source_product": True,
        },
        "normalization": {
            "mean": [1087, 1342, 1433, 2734, 1958, 1363],
            "std": [2248, 2179, 2178, 1850, 1242, 1049],
        },
        "inference": {
            "patch_size": 224,
            "stride": 112,
            "overlap_blending": "mean_softmax_probability",
        },
    }


def test_accepts_protocol_a_contract():
    value = manifest()
    assert validate_inference_manifest(value) is value


def test_accepts_protocol_b_contract():
    value = manifest()
    value["protocol"] = "B_spatial_transfer"
    value["dataset_id"] = "new_region_2026"
    value["region_id"] = "new_region"
    assert validate_inference_manifest(value) is value


def test_accepts_single_date_for_single_frame_checkpoint():
    value = manifest()
    value["images"] = [value["images"][-1]]
    value["target_date"] = value["images"][0]["date"]
    value["model_input_frames"] = 1
    assert validate_inference_manifest(value, expected_frames=1) is value


def test_rejects_single_date_for_six_frame_checkpoint():
    value = manifest()
    value["images"] = [value["images"][-1]]
    value["target_date"] = value["images"][0]["date"]
    with pytest.raises(ValueError, match="exactly 6"):
        validate_inference_manifest(value)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["images"].pop(), "exactly 6"),
        (lambda value: value.update(target_date="2027-09-21"), "final acquisition"),
        (lambda value: value["bands"].reverse(), "Band order"),
        (
            lambda value: value["radiometry"].update(offset_verified_for_source_product=False),
            "do not subtract 1000 twice",
        ),
    ],
)
def test_rejects_contract_drift(mutate, message):
    value = copy.deepcopy(manifest())
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_inference_manifest(value)

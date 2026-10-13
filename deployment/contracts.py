"""Pure-Python contracts shared by data preparation and GPU inference."""

from __future__ import annotations

import math
from datetime import date
from typing import Any


BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]
EXPECTED_MEAN = [1087, 1342, 1433, 2734, 1958, 1363]
EXPECTED_STD = [2248, 2179, 2178, 1850, 1242, 1049]
ALLOWED_PROTOCOLS = {"A_temporal_transfer", "B_spatial_transfer"}


def validate_inference_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the model-facing contract without reading raster bytes."""

    if manifest.get("schema") != "prithvi-inference-dataset-v1":
        raise ValueError("Unsupported inference dataset schema")
    if manifest.get("protocol") not in ALLOWED_PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(ALLOWED_PROTOCOLS)}")
    if not str(manifest.get("dataset_id", "")).strip():
        raise ValueError("dataset_id is required")
    if not str(manifest.get("region_id", "")).strip():
        raise ValueError("region_id is required")

    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != 6:
        raise ValueError("The P2 checkpoint requires exactly six acquisitions")
    dates = [str(spec.get("date", "")) for spec in images]
    parsed = [date.fromisoformat(value) for value in dates]
    if parsed != sorted(set(parsed)):
        raise ValueError("Acquisition dates must be unique and sorted")
    if manifest.get("target_date") != dates[-1]:
        raise ValueError("target_date must equal the final acquisition date")
    for spec in images:
        if not str(spec.get("path", "")).strip():
            raise ValueError("Every image requires a relative path")
        checksum = str(spec.get("sha256", ""))
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum.lower()):
            raise ValueError("Every image requires a lowercase SHA-256 checksum")

    if manifest.get("bands") != BAND_ORDER:
        raise ValueError(f"Band order must be {BAND_ORDER}")
    radiometry = manifest.get("radiometry", {})
    if float(radiometry.get("dn_offset", math.nan)) != -1000.0:
        raise ValueError("P2 inference requires reader DN offset -1000")
    if float(radiometry.get("scale", math.nan)) != 10000.0:
        raise ValueError("P2 inference requires scale 10000")
    if not radiometry.get("offset_verified_for_source_product", False):
        raise ValueError(
            "Confirm offset_verified_for_source_product before inference; do not subtract 1000 twice"
        )

    normalization = manifest.get("normalization", {})
    if normalization.get("mean") != EXPECTED_MEAN or normalization.get("std") != EXPECTED_STD:
        raise ValueError("Normalization differs from the completed P2 training contract")
    patch = manifest.get("inference", {})
    if int(patch.get("patch_size", 0)) != 224 or int(patch.get("stride", 0)) != 112:
        raise ValueError("P2 map inference requires patch size 224 and stride 112")
    if patch.get("overlap_blending") != "mean_softmax_probability":
        raise ValueError("Unsupported overlap blending rule")
    return manifest

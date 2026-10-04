from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.warp import transform as transform_coords

from xuanthuy_seg.contracts import atomic_json, sha256_file, stable_hash
from xuanthuy_seg.data.splits import build_split_domains
from xuanthuy_seg.data.preparation import _sample_manifest, _build_schedule, _build_validation_manifest
from xuanthuy_seg.losses.cross_entropy import class_weights_from_counts


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c["schema"] != "prithvi-xtseg-v1":
        raise ValueError("Unsupported experiment schema")
    dates = [x["date"] for x in c["images"]]
    if dates != sorted(set(dates)) or not set(c["active_dates"]) <= set(dates):
        raise ValueError("Acquisition dates must be unique, sorted, and contain active_dates")
    if c["active_dates"] != sorted(set(c["active_dates"])):
        raise ValueError("active_dates must be sorted and unique")
    if c["target_date"] != c["active_dates"][-1]:
        raise ValueError("This target-date readout requires the last active image to be the target")
    if c["bands"] != ["B02", "B03", "B04", "B8A", "B11", "B12"]:
        raise ValueError("Pretrained six-band order must be preserved")
    if c["training"]["effective_batch_size"] != c["split"]["sampling"]["batch_size"]:
        raise ValueError("Effective batch and frozen batch schedule disagree")
    m = c["model"]
    if m["decoder"] == "UNetDecoder":
        if not isinstance(m["decoder_channels"], list) or len(m["decoder_channels"]) != len(m["out_indices"]):
            raise ValueError("UNetDecoder needs one decoder channel per encoder output index")
    elif m["decoder"] != "UperNetDecoder":
        raise ValueError(f"Unsupported decoder: {m['decoder']}")
    return c


def _check_grid(src, profile):
    if (src.shape != (profile["height"], profile["width"]) or
            src.crs != profile["crs"] or src.transform != profile["transform"]):
        raise ValueError(f"Raster grid differs: {src.name}")


class Scene:
    def __init__(self, config, data_root, verify=True):
        self.config = config
        self.root = Path(data_root)
        arrays, valid, rows = [], [], []
        for spec in config["images"]:
            p = self.root / spec["path"]
            if not p.is_file():
                raise FileNotFoundError(p)
            actual = sha256_file(p)
            if verify and actual != spec["sha256"]:
                raise ValueError(f"Image checksum changed: {p.name}; create a new contract")
            with rasterio.open(p) as s:
                if not arrays:
                    self.profile = s.profile.copy()
                    if list(s.shape) != config["expected_shape"] or str(s.crs) != config["expected_crs"]:
                        raise ValueError("Image shape or CRS differs from contract")
                else:
                    _check_grid(s, self.profile)
                if s.count != 6 or list(s.descriptions) != config["bands"]:
                    raise ValueError(f"Band count/descriptions differ: {p.name}")
                if s.tags().get("acquisition_date") != spec["date"]:
                    raise ValueError(f"Acquisition metadata differs: {p.name}")
                a = s.read()
                m = (s.read_masks() > 0).all(0) & np.isfinite(a).all(0)
                arrays.append(a)
                scl_spec = config.get("cloud_masks", {}).get(spec["date"])
                if scl_spec:
                    mp = self.root / scl_spec["path"]
                    if sha256_file(mp) != scl_spec["sha256"]:
                        raise ValueError("SCL checksum differs")
                    with rasterio.open(mp) as ms:
                        _check_grid(ms, self.profile)
                        # SCL clear surface categories: dark, vegetation, bare land, water.
                        m &= np.isin(ms.read(1), [2, 4, 5, 6])
                valid.append(m)
                rows.append({"date": spec["date"], "file": p.name, "valid_pixels": int(m.sum()),
                             "sha256": actual, "cloud_mask_available": bool(scl_spec)})
        self.raw = np.stack(arrays)  # T,C,H,W; preserve uint16, normalize only crops
        self.image_valid = np.logical_and.reduce(valid)
        self.valid_by_time = np.stack(valid)
        self.active_indices = [next(i for i, x in enumerate(config["images"]) if x["date"] == d)
                               for d in config["active_dates"]]
        lp = self.root / config["label"]["path"]
        if verify and sha256_file(lp) != config["label"]["sha256"]:
            raise ValueError("Label checksum differs")
        with rasterio.open(lp) as s:
            _check_grid(s, self.profile)
            if s.count != 1:
                raise ValueError("Label raster must have one band")
            self.label = s.read(1)
        if not set(np.unique(self.label)) <= set(range(12)):
            raise ValueError("Label codes must be 0 or 1..11")
        self.common_valid = self.image_valid & np.isin(self.label, range(1, 12))
        self.domains = build_split_domains(self.label.shape, self.common_valid, config["split"])
        self.train_mask = self.common_valid & (self.domains["split_mask"] == 1)
        self.val_mask = self.common_valid & self.domains["validation_metric"]
        self.counts = np.bincount(self.label[self.train_mask], minlength=12)[1:12]
        if np.any(self.counts == 0) or not self.val_mask.any():
            raise ValueError("Empty training class or validation core")
        self.weights = class_weights_from_counts(self.counts)
        self.temporal = torch.tensor([[date.fromisoformat(d).year, date.fromisoformat(d).timetuple().tm_yday]
                                      for d in config["active_dates"]], dtype=torch.float32)
        self.audit = {"images": rows, "shape": list(self.label.shape),
                      "common_image_valid_pixels": int(self.image_valid.sum()),
                      "common_labeled_pixels": int(self.common_valid.sum()),
                      "train_pixels": int(self.train_mask.sum()), "validation_pixels": int(self.val_mask.sum()),
                      "train_class_counts": self.counts.tolist(), "class_weights": self.weights.tolist(),
                      "quality_status": config["quality_status"], "radiometry": config["radiometry"],
                      "label_reference": config["label"], "shared_input_pixels":
                      int((self.domains["train_input"] & self.domains["validation_input"]).sum()),
                      "common_validity_policy": "intersection of all six acquisitions for BOTH T=1 and T=6"}
        if not config.get("cloud_masks"):
            warnings.warn("No cloud/SCL masks: nodata validity is not cloud validity; exploratory experiment.")
        if not config["radiometry"].get("confirmed_from_export_code"):
            warnings.warn("Radiometric offset is an explicit assumption; verify export code before scientific reporting.")

    def crop(self, row, col, height=224, width=224, domain=None):
        n = self.config["split"]["patch_policy"]["size"]
        region = np.s_[row:row+height, col:col+width]
        # Invalid values filled with pretrained mean AFTER offset correction => normalized zero.
        x = self.raw[self.active_indices, :, row:row+height, col:col+width].astype(np.float32)
        x += float(self.config["radiometry"]["dn_offset"])
        # Reference constants are in scaled reflectance units (x10000), not raw offset DN.
        x *= 10000. / float(self.config["radiometry"]["scale"])
        mean = np.asarray(self.config["normalization"]["mean"], np.float32)[None, :, None, None]
        std = np.asarray(self.config["normalization"]["std"], np.float32)[None, :, None, None]
        x = (x - mean) / std
        pixel_valid = self.image_valid[region].copy()
        if domain is not None:
            pixel_valid &= domain[region]
        x[:, :, ~pixel_valid] = 0
        x = np.pad(x, ((0,0),(0,0),(0,n-height),(0,n-width)), constant_values=0)
        y = np.full((n,n), 255, np.int64)
        tvalid = self.common_valid[region] & pixel_valid
        y[:height,:width][tvalid] = self.label[region][tvalid].astype(np.int64)-1
        # Per-patch geolocation, same crop at all dates. Coordinates are latitude, longitude.
        xx, yy = self.profile["transform"] * (col + width / 2, row + height / 2)
        lon, lat = transform_coords(self.profile["crs"], "EPSG:4326", [xx], [yy])
        return {"image": torch.from_numpy(x.transpose(1,0,2,3).copy()),
                "target": torch.from_numpy(y), "temporal_coords": self.temporal.clone(),
                "location_coords": torch.tensor([lat[0],lon[0]], dtype=torch.float32)}


def prepare(scene, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    c = scene.config
    classes = {int(k):v for k,v in c["class_map"].items()}
    manifest = _sample_manifest(scene.label, scene.common_valid, scene.domains["split_mask"],
                                scene.domains["train_input"], c["split"], classes)
    schedule, support = _build_schedule(manifest, c["split"], classes)
    val = _build_validation_manifest(scene.label.shape, c["split"], scene.image_valid,
                                     scene.val_mask, scene.domains["validation_input"])
    for name, frame in [("train_manifest",manifest),("batch_schedule",schedule),
                        ("batch_support",support),("validation_manifest",val)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    for name in ["split_mask","train_input","validation_input","validation_metric"]:
        profile = scene.profile.copy(); profile.update(count=1,dtype="uint8",nodata=None,compress="deflate")
        with rasterio.open(output / f"{name}.tif", "w", **profile) as dst:
            dst.write(scene.domains[name].astype(np.uint8),1)
    artifacts = {p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.suffix in {".csv", ".tif"}}
    scene.audit.update(n_patches=len(manifest), n_batches=int(schedule.batch_index.nunique()),
                       n_validation_windows=len(val), artifact_sha256=artifacts)
    atomic_json(scene.audit, output / "DATA_AUDIT.json")
    batches = [g.sort_values("slot").manifest_index.to_numpy(np.int64)
               for _,g in schedule.groupby("batch_index", sort=True)]
    return manifest, batches, val


def stack_crops(crops, device):
    return {k:torch.stack([b[k] for b in crops]).to(device) for k in crops[0]}


def source_hash():
    return stable_hash({p.name:sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})

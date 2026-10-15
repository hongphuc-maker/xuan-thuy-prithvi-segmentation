"""Label-free inference pipeline for Protocol A and Protocol B.

This module deliberately sits outside the training package.  It consumes a
locked inference-dataset manifest plus a selected P2 checkpoint and never
opens a training split, optimizer, or label raster.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as transform_coords
from tqdm.auto import tqdm

from .contracts import BAND_ORDER, validate_inference_manifest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(path: str | Path, expected_frames: int = 6) -> dict[str, Any]:
    return validate_inference_manifest(
        json.loads(Path(path).read_text(encoding="utf-8")),
        expected_frames=expected_frames,
    )


def _check_grid(source: rasterio.io.DatasetReader, profile: dict[str, Any]) -> None:
    if (
        source.shape != (profile["height"], profile["width"])
        or source.crs != profile["crs"]
        or source.transform != profile["transform"]
    ):
        raise ValueError(f"Raster grid differs: {source.name}")


class InferenceScene:
    """Locked one- or six-date raster stack without labels or split artifacts."""

    def __init__(
        self,
        manifest_path: str | Path,
        verify: bool = True,
        expected_frames: int = 6,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.expected_frames = expected_frames
        self.manifest = load_manifest(self.manifest_path, expected_frames=expected_frames)
        self.dates = [spec["date"] for spec in self.manifest["images"]]
        arrays: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        audit_images: list[dict[str, Any]] = []

        for spec in self.manifest["images"]:
            path = (self.root / spec["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            checksum = sha256_file(path)
            if verify and checksum != spec["sha256"]:
                raise ValueError(f"Image checksum changed: {path.name}; create a new dataset contract")
            with rasterio.open(path) as source:
                if not arrays:
                    self.profile = source.profile.copy()
                    expected_shape = self.manifest.get("grid", {}).get("shape")
                    expected_crs = self.manifest.get("grid", {}).get("crs")
                    if expected_shape and list(source.shape) != list(expected_shape):
                        raise ValueError("Raster shape differs from the inference manifest")
                    if expected_crs and str(source.crs) != str(expected_crs):
                        raise ValueError("Raster CRS differs from the inference manifest")
                else:
                    _check_grid(source, self.profile)
                if source.count != 6 or list(source.descriptions) != BAND_ORDER:
                    raise ValueError(f"Band count/descriptions differ: {path.name}")
                if source.tags().get("acquisition_date") != spec["date"]:
                    raise ValueError(f"acquisition_date tag differs: {path.name}")
                array = source.read()
                valid = (source.read_masks() > 0).all(0) & np.isfinite(array).all(0)
                arrays.append(array)
                valid_masks.append(valid)
                audit_images.append(
                    {
                        "date": spec["date"],
                        "file": path.name,
                        "sha256": checksum,
                        "valid_pixels": int(valid.sum()),
                    }
                )

        self.raw = np.stack(arrays)
        self.valid_by_time = np.stack(valid_masks)
        self.image_valid = np.logical_and.reduce(valid_masks)
        self.temporal = torch.tensor(
            [[d.year, d.timetuple().tm_yday] for d in map(date.fromisoformat, self.dates)],
            dtype=torch.float32,
        )
        self.audit = {
            "dataset_id": self.manifest["dataset_id"],
            "protocol": self.manifest["protocol"],
            "shape": list(self.image_valid.shape),
            "common_valid_pixels": int(self.image_valid.sum()),
            "images": audit_images,
            "manifest_sha256": sha256_file(self.manifest_path),
        }
        if not self.image_valid.any():
            raise ValueError("No pixel is valid across the locked input acquisition(s)")

    @property
    def patch_size(self) -> int:
        return int(self.manifest["inference"]["patch_size"])

    @property
    def stride(self) -> int:
        return int(self.manifest["inference"]["stride"])

    def crop(self, row: int, col: int, height: int | None = None, width: int | None = None):
        size = self.patch_size
        height = min(size, self.profile["height"] - row) if height is None else int(height)
        width = min(size, self.profile["width"] - col) if width is None else int(width)
        region = np.s_[row : row + height, col : col + width]
        x = self.raw[:, :, row : row + height, col : col + width].astype(np.float32)
        radiometry = self.manifest["radiometry"]
        x += float(radiometry["dn_offset"])
        x *= 10000.0 / float(radiometry["scale"])
        normalization = self.manifest["normalization"]
        mean = np.asarray(normalization["mean"], np.float32)[None, :, None, None]
        std = np.asarray(normalization["std"], np.float32)[None, :, None, None]
        x = (x - mean) / std
        valid = self.image_valid[region].copy()
        x[:, :, ~valid] = 0
        x = np.pad(x, ((0, 0), (0, 0), (0, size - height), (0, size - width)))
        xx, yy = self.profile["transform"] * (col + width / 2, row + height / 2)
        longitude, latitude = transform_coords(
            self.profile["crs"], "EPSG:4326", [xx], [yy]
        )
        return {
            "image": torch.from_numpy(x.transpose(1, 0, 2, 3).copy()),
            "temporal_coords": self.temporal.clone(),
            "location_coords": torch.tensor([latitude[0], longitude[0]], dtype=torch.float32),
            "height": height,
            "width": width,
            "row": row,
            "col": col,
        }


def _stack_crops(crops: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([crop["image"] for crop in crops]).to(device, non_blocking=True),
        "temporal_coords": torch.stack([crop["temporal_coords"] for crop in crops]).to(
            device, non_blocking=True
        ),
        "location_coords": torch.stack([crop["location_coords"] for crop in crops]).to(
            device, non_blocking=True
        ),
    }


def _starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def inference_windows(scene: InferenceScene) -> list[dict[str, int]]:
    height, width = scene.image_valid.shape
    size, stride = scene.patch_size, scene.stride
    windows = []
    for row in _starts(height, size, stride):
        for col in _starts(width, size, stride):
            window_valid = scene.image_valid[row : row + size, col : col + size]
            if window_valid.any():
                windows.append(
                    {
                        "row": row,
                        "col": col,
                        "height": min(size, height - row),
                        "width": min(size, width - col),
                    }
                )
    return windows


def load_wce_model(
    base_config_path: str | Path,
    manifest: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device | None = None,
):
    """Build the frame-matched WCE architecture and load its selected checkpoint."""

    base_config = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    if base_config.get("schema") != "prithvi-xtseg-v1":
        raise ValueError("The deployment path supports the completed P2 WCE contract only")
    if base_config.get("bands") != BAND_ORDER:
        raise ValueError("Base training configuration uses a different band contract")
    expected_frames = len(base_config.get("active_dates", []))
    validate_inference_manifest(manifest, expected_frames=expected_frames)
    config = copy.deepcopy(base_config)
    config["active_dates"] = [spec["date"] for spec in manifest["images"]]
    config["target_date"] = manifest["target_date"]
    if len(config["active_dates"]) != expected_frames:
        raise ValueError("Inference frames differ from the selected checkpoint architecture")

    # The notebook must put the frozen reference/wce_completed package on sys.path.
    from prithvi_xtseg.model import build_model

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, pretrained=False).to(device)
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in payload:
        raise ValueError("Checkpoint payload has no model state")
    model.load_state_dict(payload["model"], strict=True)
    model.pretrained_provenance = payload.get("pretrained_provenance", {})
    model.eval()
    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_method_hash": payload.get("method_hash"),
        "checkpoint_summary": payload.get("summary"),
        "base_config_sha256": sha256_file(base_config_path),
        "active_dates": config["active_dates"],
        "target_date": config["target_date"],
    }
    return model, config, provenance, device


def _predict_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(
        batch["image"],
        temporal_coords=batch["temporal_coords"],
        location_coords=batch["location_coords"],
    ).output
    expected = (batch["image"].shape[0], 11, *batch["image"].shape[-2:])
    if tuple(output.shape) != expected:
        raise ValueError(f"Wrong segmentation shape {tuple(output.shape)}; expected {expected}")
    return output


@torch.inference_mode()
def probability_mosaic(
    model,
    scene: InferenceScene,
    device: torch.device,
    windows: Iterable[dict[str, int]] | None = None,
    progress: bool = True,
):
    windows = list(inference_windows(scene) if windows is None else windows)
    height, width = scene.image_valid.shape
    probabilities = np.zeros((11, height, width), np.float32)
    counts = np.zeros((height, width), np.uint16)
    for window in tqdm(windows, desc="Probability mosaic", disable=not progress):
        crop = scene.crop(**window)
        batch = _stack_crops([crop], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = _predict_logits(model, batch)
        probability = logits.float().softmax(1)[0, :, : crop["height"], : crop["width"]]
        probability = probability.cpu().numpy()
        row, col, h, w = crop["row"], crop["col"], crop["height"], crop["width"]
        valid = scene.image_valid[row : row + h, col : col + w]
        probabilities[:, row : row + h, col : col + w] += probability * valid[None]
        counts[row : row + h, col : col + w] += valid.astype(np.uint16)
    np.divide(probabilities, counts[None], out=probabilities, where=counts[None] > 0)
    return probabilities, counts


def export_probability_map(
    model,
    scene: InferenceScene,
    checkpoint_provenance: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    """Run tiled inference and write a sealed, label-free GeoTIFF product."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = {
        "manifest_sha256": sha256_file(scene.manifest_path),
        "checkpoint_sha256": checkpoint_provenance["checkpoint_sha256"],
        "patch_size": scene.patch_size,
        "stride": scene.stride,
        "overlap_blending": "mean_softmax_probability",
    }
    inference_hash = _stable_hash(method)
    marker_path = output_dir / "INFERENCE_COMPLETE.json"
    if marker_path.exists():
        previous = json.loads(marker_path.read_text(encoding="utf-8"))
        if previous.get("inference_hash") != inference_hash:
            raise ValueError("Output directory is sealed for a different dataset or checkpoint")
        for filename, checksum in previous["artifacts"].items():
            if sha256_file(output_dir / filename) != checksum:
                raise ValueError(f"Sealed inference artifact changed: {filename}")
        return previous
    existing_names = {path.name for path in output_dir.iterdir()}
    unexpected = existing_names - {"T4_BENCHMARK.json"}
    if unexpected:
        raise ValueError(
            "Use a new output directory after an interrupted or different inference run; "
            f"unexpected existing files: {sorted(unexpected)}"
        )

    _atomic_json(
        {
            "inference_hash": inference_hash,
            "method": method,
            "dataset": scene.audit,
            "checkpoint": checkpoint_provenance,
        },
        output_dir / "INFERENCE_LOCK.json",
    )
    started = time.perf_counter()
    probabilities, counts = probability_mosaic(model, scene, device, progress=progress)
    valid = scene.image_valid & (counts > 0)
    if not np.all(counts[scene.image_valid] > 0):
        raise ValueError("Map has uncovered valid pixels")
    classes = np.zeros(valid.shape, np.uint8)
    classes[valid] = probabilities[:, valid].argmax(0).astype(np.uint8) + 1
    confidence = np.full(valid.shape, np.nan, np.float32)
    confidence[valid] = probabilities[:, valid].max(0)
    entropy = np.full(valid.shape, np.nan, np.float32)
    entropy[valid] = -(
        probabilities[:, valid] * np.log(np.maximum(probabilities[:, valid], 1e-12))
    ).sum(0) / np.log(11)

    date_token = scene.manifest["target_date"].replace("-", "")
    arrays = {
        f"classes_{date_token}.tif": classes,
        f"confidence_{date_token}_uncalibrated.tif": confidence,
        f"entropy_{date_token}_uncalibrated.tif": entropy,
        "common_validity.tif": valid.astype(np.uint8),
    }
    for filename, array in arrays.items():
        path = output_dir / filename
        temporary = path.with_name(path.stem + ".tmp.tif")
        nodata = np.nan if np.issubdtype(array.dtype, np.floating) else 0
        profile = scene.profile.copy()
        profile.update(count=1, dtype=str(array.dtype), nodata=nodata, compress="deflate")
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(array, 1)
            destination.update_tags(
                dataset_id=scene.manifest["dataset_id"],
                protocol=scene.manifest["protocol"],
                target_date=scene.manifest["target_date"],
                input_dates=",".join(scene.dates),
                checkpoint_sha256=checkpoint_provenance["checkpoint_sha256"],
                inference_hash=inference_hash,
                evaluation_status="unscored inference product",
            )
        os.replace(temporary, path)

    marker = {
        "inference_hash": inference_hash,
        "dataset_id": scene.manifest["dataset_id"],
        "protocol": scene.manifest["protocol"],
        "target_date": scene.manifest["target_date"],
        "input_dates": scene.dates,
        "valid_pixels": int(valid.sum()),
        "windows": len(inference_windows(scene)),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": checkpoint_provenance["checkpoint_sha256"],
        "artifacts": {filename: sha256_file(output_dir / filename) for filename in arrays},
        "evaluation_status": (
            "No date-matched ground truth was used. This product is qualitative until a separate "
            "evaluation protocol is supplied."
        ),
    }
    _atomic_json(marker, marker_path)
    return marker


@torch.inference_mode()
def benchmark_inference(
    model,
    scene: InferenceScene,
    device: torch.device,
    sample_tiles: int = 30,
    warmup_tiles: int = 5,
) -> dict[str, Any]:
    """Measure compute and end-to-end tile latency on the active CUDA device."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires a CUDA runtime")
    windows = inference_windows(scene)
    if not windows:
        raise ValueError("No valid inference windows")
    sample = windows[: min(len(windows), warmup_tiles + sample_tiles)]
    if len(sample) <= warmup_tiles:
        raise ValueError("Not enough valid tiles for the requested warm-up")

    for window in sample[:warmup_tiles]:
        batch = _stack_crops([scene.crop(**window)], device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _predict_logits(model, batch).float().softmax(1)
    torch.cuda.synchronize()

    compute_ms: list[float] = []
    end_to_end_ms: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for window in sample[warmup_tiles:]:
        wall_start = time.perf_counter()
        crop = scene.crop(**window)
        batch = _stack_crops([crop], device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probability = _predict_logits(model, batch).float().softmax(1)
        end_event.record()
        torch.cuda.synchronize()
        _ = probability.shape
        compute_ms.append(float(start_event.elapsed_time(end_event)))
        end_to_end_ms.append((time.perf_counter() - wall_start) * 1000.0)

    compute = np.asarray(compute_ms)
    end_to_end = np.asarray(end_to_end_ms)
    total_windows = len(windows)
    patch_pixels = scene.patch_size * scene.patch_size
    median_e2e_seconds = float(np.median(end_to_end) / 1000.0)
    return {
        "benchmark_scope": "batch-1 tile latency; map time below is an estimate, not a measured full-map run",
        "device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "sample_tiles": int(len(end_to_end)),
        "warmup_tiles": int(warmup_tiles),
        "patch_size": scene.patch_size,
        "stride": scene.stride,
        "total_map_windows": total_windows,
        "compute_median_ms": float(np.median(compute)),
        "compute_p95_ms": float(np.percentile(compute, 95)),
        "end_to_end_median_ms": float(np.median(end_to_end)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
        "end_to_end_tiles_per_second": float(1000.0 / np.median(end_to_end)),
        "end_to_end_megapixels_per_second": float(
            patch_pixels / 1_000_000.0 * (1000.0 / np.median(end_to_end))
        ),
        "estimated_serial_map_seconds": total_windows * median_e2e_seconds,
        "peak_cuda_memory_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3),
    }

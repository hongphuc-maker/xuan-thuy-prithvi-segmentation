from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm.auto import tqdm

from xuanthuy_seg.contracts import atomic_json, sha256_file

from .data import stack_crops
from .dmi import numpy_joint_and_diagnostics
from .model import predict_logits
from .permutation import align_probabilities, confusion_from_predictions, learn_class_permutation


def metrics(confusion: np.ndarray, classes: dict[str, str]) -> tuple[dict[str, object], pd.DataFrame]:
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(1).astype(float)
    predicted = confusion.sum(0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(11), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(11), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(11), where=support + predicted > 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(11), where=union > 0)
    frame = pd.DataFrame(
        {
            "code": range(1, 12),
            "class": [classes[str(i)] for i in range(1, 12)],
            "support": support.astype(int),
            "predicted": predicted.astype(int),
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "IoU": iou,
        }
    )
    summary = {
        "OA": float(tp.sum() / max(support.sum(), 1)),
        "macro_F1_11": float(f1.mean()),
        "macro_IoU_11": float(iou.mean()),
        "predicted_classes": int((predicted > 0).sum()),
        "n_pixels": int(support.sum()),
    }
    return summary, frame


def starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    result = list(range(0, length - size + 1, stride))
    if result[-1] != length - size:
        result.append(length - size)
    return result


@torch.inference_mode()
def mosaic(
    model,
    scene,
    windows,
    device,
    domain=None,
    progress: bool = True,
    channel_order: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()
    height, width = scene.label.shape
    accumulator = np.zeros((11, height, width), np.float32)
    counts = np.zeros((height, width), np.uint16)
    try:
        for window in tqdm(windows, desc="Probability mosaic", disable=not progress):
            row, col, crop_height, crop_width = [
                int(window[key]) for key in ("row", "col", "height", "width")
            ]
            crop = scene.crop(row, col, crop_height, crop_width, domain=domain)
            batch = stack_crops([crop], device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = predict_logits(model, batch)
            probability = logits.float().softmax(1)[0, :, :crop_height, :crop_width].cpu().numpy()
            if channel_order is not None:
                probability = probability[np.asarray(channel_order, dtype=np.int64)]
            valid = scene.image_valid[row : row + crop_height, col : col + crop_width].copy()
            if domain is not None:
                valid &= domain[row : row + crop_height, col : col + crop_width]
            accumulator[:, row : row + crop_height, col : col + crop_width] += probability * valid[None]
            counts[row : row + crop_height, col : col + crop_width] += valid.astype(np.uint16)
    finally:
        model.train(was_training)
    np.divide(accumulator, counts[None], out=accumulator, where=counts[None] > 0)
    return accumulator, counts


def learn_train_permutation(model, scene, manifest: pd.DataFrame, indices: list[int], device):
    windows = []
    for index in indices:
        row = manifest.iloc[int(index)]
        windows.append(
            {
                "row": int(row.row),
                "col": int(row.col),
                "height": int(getattr(row, "height", 224)),
                "width": int(getattr(row, "width", 224)),
            }
        )
    probability, counts = mosaic(
        model,
        scene,
        windows,
        device,
        domain=scene.domains["train_input"],
        progress=False,
    )
    mask = scene.train_mask & (counts > 0)
    if not mask.any():
        raise ValueError("Train-only permutation panel covers zero labeled pixels")
    truth = scene.label[mask].astype(np.int64) - 1
    raw_prediction = probability[:, mask].argmax(0)
    confusion = confusion_from_predictions(truth, raw_prediction, classes=11)
    permutation = learn_class_permutation(confusion)
    permutation["n_unique_train_pixels"] = int(mask.sum())
    permutation["panel_manifest_indices"] = [int(value) for value in indices]
    return permutation


def validation(model, scene, val_manifest, manifest, permutation_indices, device):
    permutation = learn_train_permutation(model, scene, manifest, permutation_indices, device)
    raw_probability, counts = mosaic(
        model,
        scene,
        val_manifest.to_dict("records"),
        device,
        domain=scene.domains["validation_input"],
        progress=False,
    )
    mask = scene.val_mask
    if not np.all(counts[mask] > 0):
        raise ValueError("Validation core has uncovered pixels")
    truth = scene.label[mask].astype(np.int64) - 1
    aligned_probability = align_probabilities(raw_probability, permutation)
    prediction = aligned_probability[:, mask].argmax(0)
    confusion = confusion_from_predictions(truth, prediction, classes=11)
    summary, frame = metrics(confusion, scene.config["class_map"])

    raw_flat = raw_probability[:, mask].T
    _, dmi = numpy_joint_and_diagnostics(raw_flat, truth)
    summary.update(
        validation_DMI_loss=dmi.loss,
        validation_DMI_sign=dmi.sign,
        validation_DMI_rank=dmi.rank,
        validation_DMI_min_singular_value=dmi.min_singular_value,
        validation_DMI_max_singular_value=dmi.max_singular_value,
        validation_DMI_condition_number=dmi.condition_number,
        validation_DMI_finite=dmi.finite_loss,
        permutation_train_macro_recall=float(permutation["train_panel_macro_recall"]),
    )
    gate = scene.config["acceptance"]
    passed = summary["predicted_classes"] == gate["required_predicted_classes"]
    passed &= all(
        frame.loc[frame.code == int(code), "recall"].iloc[0] >= threshold
        for code, threshold in gate["minimum_recall_by_code"].items()
    )
    summary["acceptance_passed"] = bool(passed)
    summary["validation_DMI_checkpoint_eligible"] = bool(
        dmi.finite_loss
        and dmi.sign != 0.0
        and dmi.rank == 11
        and np.isfinite(dmi.condition_number)
    )
    return summary, frame, permutation


def create_map(model, scene, run_root, checkpoint_path, checkpoint_payload, device):
    run_root = Path(run_root)
    output = run_root / "map"
    output.mkdir(exist_ok=True)
    lock = json.loads((run_root / "RUN_LOCK.json").read_text())
    checkpoint_hash = sha256_file(checkpoint_path)
    permutation = checkpoint_payload["permutation"]
    order = [int(value) for value in permutation["true_class_to_output_channel"]]
    marker = output / "MAP_COMPLETE.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["method_hash"] != lock["method_hash"] or previous["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Map already sealed for a different checkpoint; use a new run")
        for name, digest in previous["artifacts"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("Sealed map artifact changed")
        return previous

    height, width = scene.label.shape
    size = scene.config["split"]["patch_policy"]["size"]
    stride = scene.config["split"]["patch_policy"]["inference_stride"]
    windows = [
        {
            "row": row,
            "col": col,
            "height": min(size, height - row),
            "width": min(size, width - col),
        }
        for row in starts(height, size, stride)
        for col in starts(width, size, stride)
        if scene.image_valid[row : row + size, col : col + size].any()
    ]
    probability, counts = mosaic(
        model, scene, windows, device, channel_order=order
    )
    valid = scene.image_valid & (counts > 0)
    if not np.all(counts[scene.image_valid] > 0):
        raise ValueError("Map has uncovered valid pixels")
    code = np.zeros((height, width), np.uint8)
    code[valid] = probability[:, valid].argmax(0).astype(np.uint8) + 1
    confidence = np.full((height, width), np.nan, np.float32)
    confidence[valid] = probability[:, valid].max(0)
    entropy = np.full((height, width), np.nan, np.float32)
    entropy[valid] = -(
        probability[:, valid] * np.log(np.maximum(probability[:, valid], 1e-12))
    ).sum(0) / np.log(11)
    arrays = {
        "classes_sep21_2026.tif": code,
        "max_softmax_uncalibrated.tif": confidence,
        "entropy_uncalibrated.tif": entropy,
        "common_validity.tif": valid.astype(np.uint8),
    }
    permutation_json = json.dumps(order, separators=(",", ":"))
    for name, array in arrays.items():
        path = output / name
        temporary = path.with_name(path.stem + ".tmp.tif")
        nodata = np.nan if np.issubdtype(array.dtype, np.floating) else 0
        profile = scene.profile.copy()
        profile.update(count=1, dtype=str(array.dtype), nodata=nodata, compress="deflate")
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(array, 1)
            destination.update_tags(
                target_date=scene.config["target_date"],
                method_hash=lock["method_hash"],
                input_dates=",".join(scene.config["active_dates"]),
                label_status=scene.config["label"]["status"],
                quality_status=scene.config["quality_status"],
                true_class_to_output_channel=permutation_json,
            )
        temporary.replace(path)
    marker_data = {
        "method_hash": lock["method_hash"],
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_summary": checkpoint_payload["summary"],
        "permutation": permutation,
        "target_date": scene.config["target_date"],
        "valid_pixels": int(valid.sum()),
        "artifacts": {name: sha256_file(output / name) for name in arrays},
        "note": "Target September map; evaluated only against historical labels, not date-matched September ground truth",
    }
    atomic_json(marker_data, marker)
    return marker_data


def evaluate_reference_points(scene, run_root, points_root, confirm=False):
    if not confirm:
        raise ValueError("Enable this cell only after checkpoint/map selection is final")
    import geopandas as gpd
    import tempfile
    import zipfile

    run_root = Path(run_root)
    output = run_root / "reference_point_agreement"
    output.mkdir(exist_ok=True)
    marker = run_root / "map" / "MAP_COMPLETE.json"
    seal = json.loads(marker.read_text())
    map_path = run_root / "map" / "classes_sep21_2026.tif"
    if sha256_file(map_path) != seal["artifacts"][map_path.name]:
        raise ValueError("Map checksum changed")
    specification = scene.config["independent_points"]
    zip_path = Path(points_root) / specification["path"]
    if sha256_file(zip_path) != specification["sha256"]:
        raise ValueError("Point ZIP checksum differs")
    completion = output / "AGREEMENT_COMPLETE.json"
    if completion.exists():
        previous = json.loads(completion.read_text())
        if previous["map_sha256"] != sha256_file(map_path):
            raise ValueError("Evaluation map changed")
        return previous
    by_name = {value.casefold(): int(key) for key, value in scene.config["class_map"].items()}
    records = []
    with rasterio.open(map_path) as source, tempfile.TemporaryDirectory() as temporary_directory:
        with zipfile.ZipFile(zip_path) as archive:
            for entry in archive.infolist():
                destination = (Path(temporary_directory) / entry.filename).resolve()
                if Path(temporary_directory).resolve() not in destination.parents:
                    raise ValueError("Unsafe ZIP path")
            archive.extractall(temporary_directory)
        files = sorted(Path(temporary_directory).rglob("*.shp"))
        if not files:
            raise ValueError("No point shapefiles")
        for file in files:
            if file.stem.casefold() not in by_name:
                raise ValueError(f"Unknown class layer {file.stem}")
            frame = gpd.read_file(file)
            if frame.crs is None:
                raise ValueError("Point layer missing CRS")
            frame = frame.to_crs(source.crs)
            for index, point in frame.geometry.items():
                if point is None or point.geom_type != "Point":
                    raise ValueError("Point geometry required")
                row, col = source.index(point.x, point.y)
                inside = 0 <= row < source.height and 0 <= col < source.width
                prediction = int(next(source.sample([(point.x, point.y)]))[0]) if inside else 0
                records.append(
                    {
                        "layer": file.stem,
                        "id": str(index),
                        "truth": by_name[file.stem.casefold()],
                        "pred": prediction,
                        "row": row,
                        "col": col,
                        "usable": inside and 1 <= prediction <= 11,
                    }
                )
    frame = pd.DataFrame(records)
    usable = frame[frame.usable]
    if usable.empty:
        raise ValueError("No usable reference points")
    confusion = np.bincount(
        (usable.truth.to_numpy() - 1) * 11 + usable.pred.to_numpy() - 1,
        minlength=121,
    ).reshape(11, 11)
    summary, per_class = metrics(confusion, scene.config["class_map"])
    summary["macro_F1_represented_classes"] = float(per_class.loc[per_class.support > 0, "F1"].mean())
    summary.update(
        input_points=len(frame),
        excluded_points=len(frame) - len(usable),
        interpretation="Agreement with previously used reference points; September accuracy not established",
        map_sha256=sha256_file(map_path),
        method_hash=seal["method_hash"],
    )
    frame.to_csv(output / "points.csv", index=False)
    per_class.to_csv(output / "per_class.csv", index=False)
    np.savetxt(output / "confusion.csv", confusion, fmt="%d", delimiter=",")
    summary["artifacts"] = {
        name: sha256_file(output / name)
        for name in ["points.csv", "per_class.csv", "confusion.csv"]
    }
    atomic_json(summary, completion)
    return summary

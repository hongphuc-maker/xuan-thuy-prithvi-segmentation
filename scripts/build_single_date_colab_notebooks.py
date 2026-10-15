"""Generate the isolated T=1 training, data-preparation, and inference notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )


REPO_SETUP = r'''from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
from pathlib import Path

REPO_URL = 'https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation.git'
REPO_REF = 'main'  # Replace with the approved commit SHA before the scientific run.
REPO_ROOT = Path('/content/xuan-thuy-prithvi-segmentation')

if not (REPO_ROOT / '.git').exists():
    subprocess.check_call(['git', 'clone', '--filter=blob:none', '--no-checkout', REPO_URL, str(REPO_ROOT)])
else:
    dirty = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise RuntimeError('The Colab repository copy has local changes; use a fresh runtime or review them.')

subprocess.check_call(['git', '-C', str(REPO_ROOT), 'fetch', '--depth=1', 'origin', REPO_REF])
subprocess.check_call(['git', '-C', str(REPO_ROOT), 'checkout', '--detach', 'FETCH_HEAD'])

REPO_COMMIT = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
print('Repository commit:', REPO_COMMIT)
'''


FULL_ENV = r'''import importlib.metadata
import os
import signal

ENV_MARKER = Path('/content/.prithvi_p3_single_env_v1')
if not ENV_MARKER.exists():
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        'numpy==2.3.5', 'terratorch==1.2.12', 'torchgeo==0.9.0', 'timm==1.0.30',
        'segmentation-models-pytorch==0.5.0', 'huggingface-hub==1.33.0',
        'einops==0.8.2', 'rasterio==1.5.2', 'pandas==2.2.3',
        'matplotlib>=3.8,<4', 'PyYAML>=6,<7', 'geopandas>=1,<2', 'tqdm>=4.66,<5',
        'git+https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation.git@0775770ae87b83b7d2982ed727226128685f3fba'
    ])
    ENV_MARKER.write_text('installed; restart required', encoding='utf-8')
    print('Environment installed. Colab will restart now; then run all cells from the top again.')
    os.kill(os.getpid(), signal.SIGKILL)

P3_SOURCE = REPO_ROOT / 'reference' / 'wce_completed'
sys.path.insert(0, str(P3_SOURCE))
sys.path.insert(0, str(REPO_ROOT))
print({name: importlib.metadata.version(name) for name in [
    'numpy', 'torch', 'terratorch', 'torchgeo', 'timm', 'rasterio', 'pandas'
]})
print('P3 source:', P3_SOURCE)
'''


def build_training() -> None:
    write(
        "11_train_p3_single_wce.ipynb",
        [
            markdown(
                """# P3 single-date training: Prithvi + U-Net + WCE

Notebook này fine-tune nhánh **T=1** trên ảnh Xuân Thủy ngày 21/09/2026.
Encoder khởi tạo từ Prithvi-EO-2.0-300M-TL chính thức; U-Net decoder và head
khởi tạo mới. Notebook không lấy trọng số P2 T=6, không tạo map và không dùng
điểm kiểm chứng để chọn checkpoint.

P3 giữ nguyên split, guard, patch schedule, WCE, seed, optimizer và tiêu chí
validation macro-F1 của P2. Dữ liệu sáu ngày vẫn được audit để tái sử dụng đúng
common-validity/split evidence; tensor đưa vào model chỉ chứa ngày 21/09."""
            ),
            markdown("## 1. Mount Drive và checkout mã nguồn"),
            code(REPO_SETUP),
            markdown(
                """## 2. Cài môi trường

Lần đầu cell này hoàn tất cài đặt, runtime sẽ tự restart để tránh trộn hai phiên
bản NumPy trong RAM. Sau restart, chạy lại notebook từ đầu; marker trong
`/content` khiến cell không cài lại lần nữa."""
            ),
            code(FULL_ENV),
            markdown("## 3. Khóa config, dữ liệu và run directory riêng"),
            code(
                r'''from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/Prithvi_Colab')
DATA_ROOT = DRIVE_ROOT / 'data' / 'training_p2'
RUN_ID = 'P3_single_sep21_UNet_WCE_REPRO'
RUN_ROOT = DRIVE_ROOT / 'runs' / RUN_ID
CONFIG_PATH = REPO_ROOT / 'configs' / 'experiments' / 'p3_prithvi_unet_wce_single.json'

assert CONFIG_PATH.is_file(), CONFIG_PATH
assert DATA_ROOT.is_dir(), (
    f'{DATA_ROOT} does not exist. Point DATA_ROOT to the folder containing the '
    'six locked training GeoTIFFs and rasterized_vector_categorized.tif.'
)
print('DATA_ROOT:', DATA_ROOT)
print('RUN_ROOT:', RUN_ROOT)
print('CONFIG_PATH:', CONFIG_PATH)
'''
            ),
            markdown("## 4. Audit dữ liệu và xác nhận T=1"),
            code(
                r'''import json
from IPython.display import display
from prithvi_xtseg.training import open_experiment

experiment = open_experiment(CONFIG_PATH, DATA_ROOT, RUN_ROOT)
display(experiment.scene.audit)
assert experiment.config['active_dates'] == ['2026-09-21']
assert experiment.scene.audit['shared_input_pixels'] == 0
print('Model active dates:', experiment.config['active_dates'])
print('Train patches:', len(experiment.manifest))
print('Validation windows:', len(experiment.val))
'''
            ),
            markdown("## 5. GPU smoke test trước optimizer update"),
            code(
                r'''import torch

if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU')
model = experiment.load_model(pretrained=not (RUN_ROOT / 'latest_resume.pt').exists())
smoke = experiment.smoke_test()
display(smoke)
assert smoke['input_shape'] == [1, 6, 1, 224, 224]
assert smoke['output_shape'] == [1, 11, 224, 224]
'''
            ),
            markdown(
                """## 6. Train hoặc resume P3

Sau khi kiểm tra `RUN_ROOT`, audit và smoke test, đổi `RUN_TRAINING=True`.
`latest_resume.pt` chỉ resume chính run P3 này; không đọc checkpoint P2."""
            ),
            code(
                r'''RUN_TRAINING = False

if not RUN_TRAINING:
    print('Training is disabled. Review the audit, then set RUN_TRAINING=True.')
else:
    training_result = experiment.train()
    display(training_result)
'''
            ),
            markdown("## 7. Đọc model-selection artifacts, không tạo map"),
            code(
                r'''for filename in [
    'TRAINING_PROGRESS.json', 'TRAINING_COMPLETE.json',
    'validation_history.csv', 'latest_validation_per_class.csv'
]:
    path = RUN_ROOT / filename
    if path.exists():
        print('\n', filename)
        if path.suffix == '.json':
            display(json.loads(path.read_text()))
        else:
            import pandas as pd
            display(pd.read_csv(path).tail(20))

for checkpoint in ['best_accepted.pt', 'best_any.pt', 'latest_resume.pt']:
    path = RUN_ROOT / checkpoint
    print(checkpoint, 'exists=' + str(path.exists()))
print('This notebook stops before map inference by design.')
'''
            ),
        ],
    )


def build_preparation() -> None:
    write(
        "21_prepare_single_date_inference_dataset.ipynb",
        [
            markdown(
                """# Prepare one Sentinel-2 acquisition for P3 inference

Notebook này audit và stage đúng **một** GeoTIFF sáu band cho checkpoint P3 T=1.
Nó không train, không load checkpoint và không sử dụng nhãn. Mặc định bên dưới
là ảnh Thái Thụy 21/09/2026 để so sánh trực tiếp P3 T=1 với P2 T=6."""
            ),
            markdown("## 1. Mount Drive và cài thư viện kiểm tra raster"),
            code(
                r'''from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
from pathlib import Path
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'numpy==2.3.5', 'rasterio==1.5.2', 'pandas==2.2.3', 'matplotlib>=3.8,<4'
])
print('If NumPy was replaced in this runtime, restart the session once before continuing.')
'''
            ),
            markdown("## 2. Khóa dataset single-date"),
            code(
                r'''from pathlib import Path

PROTOCOL = 'B_spatial_transfer'
PROTOCOL_FOLDER = 'protocol_B'
DATASET_ID = 'thai_thuy_2026_sep21_single'
REGION_ID = 'thai_thuy'
SOURCE_CONTAINER = Path(
    '/content/drive/MyDrive/Prithvi_Colab/data/sentinel2_exports/'
    'Sentinel2_ThaiThuy_2026_6bands_10m'
)
SOURCE_IMAGE = {
    'date': '2026-09-21',
    'filename': 'Sentinel2_2026-09-21_S2C_MSIL2A_20260921T_HSTDNNTHAITHUY_6bands_10m.tif',
    'stac_item_id': 'S2C_MSIL2A_20260921T031521_R118_T48QXH_20260921T082912',
}
TARGET_DATE = SOURCE_IMAGE['date']
OUTPUT_ROOT = (
    Path('/content/drive/MyDrive/Prithvi_Colab/data/inference_single')
    / PROTOCOL_FOLDER / DATASET_ID
)

assert PROTOCOL in {'A_temporal_transfer', 'B_spatial_transfer'}
assert Path(DATASET_ID).name == DATASET_ID
assert 'mosaic' not in SOURCE_IMAGE['filename'].lower()
print('Source:', SOURCE_CONTAINER)
print('Output:', OUTPUT_ROOT)
'''
            ),
            markdown("## 3. Materialize source từ folder hoặc ZIP"),
            code(
                r'''import hashlib
import zipfile

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

filename = SOURCE_IMAGE['filename']
if SOURCE_CONTAINER.is_dir():
    source_path = SOURCE_CONTAINER / filename
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
elif SOURCE_CONTAINER.is_file() and zipfile.is_zipfile(SOURCE_CONTAINER):
    staging = Path('/content') / f'sentinel2_single_{DATASET_ID}'
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_CONTAINER) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == filename]
        if len(matches) != 1:
            raise ValueError(f'Expected one ZIP member named {filename}, found {len(matches)}')
        source_path = staging / filename
        with archive.open(matches[0]) as source, open(source_path, 'wb') as target:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                target.write(block)
else:
    raise FileNotFoundError('SOURCE_CONTAINER must be a folder or ZIP')

source_sha256 = sha256_file(source_path)
print(source_path.name, source_sha256)
'''
            ),
            markdown("## 4. Audit và stage GeoTIFF không đổi DN"),
            code(
                r'''import json
import numpy as np
import rasterio

BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
rasters_dir = OUTPUT_ROOT / 'rasters'
rasters_dir.mkdir(parents=True, exist_ok=True)
output_path = rasters_dir / f"Sentinel2_{TARGET_DATE}_{REGION_ID}_6bands_10m.tif"

with rasterio.open(source_path) as source:
    if source.count != 6 or list(source.descriptions) != BANDS:
        raise ValueError('Wrong six-band contract')
    if source.tags().get('acquisition_date') != TARGET_DATE:
        raise ValueError('acquisition_date tag differs')
    if source.tags().get('source') != 'Microsoft Planetary Computer - Sentinel-2 L2A':
        raise ValueError('Unexpected source tag')
    if source.crs is None or not source.crs.is_projected:
        raise ValueError('A projected CRS is required')
    if not (
        np.isclose(abs(source.transform.a), 10.0)
        and np.isclose(abs(source.transform.e), 10.0)
        and np.isclose(source.transform.b, 0.0)
        and np.isclose(source.transform.d, 0.0)
    ):
        raise ValueError('GeoTIFF is not on a north-up 10 m grid')
    if list(source.dtypes) != ['uint16'] * 6 or source.nodata != 0:
        raise ValueError('Expected uint16 raw DN with nodata=0')
    array = source.read()
    valid = (source.read_masks() > 0).all(0) & np.isfinite(array).all(0)
    if not valid.any():
        raise ValueError('No valid pixel')
    array[:, ~valid] = 0
    profile = source.profile.copy()
    source_tags = source.tags()
    height, width = source.height, source.width
    target_crs = str(source.crs)
    transform = source.transform

if output_path.exists():
    with rasterio.open(output_path) as existing:
        if existing.tags().get('source_sha256') != source_sha256:
            raise FileExistsError('Existing output uses different source bytes; choose a new DATASET_ID')
else:
    profile.update(compress='deflate', tiled=True, nodata=0)
    output_tags = dict(source_tags)
    output_tags.update(
        acquisition_date=TARGET_DATE,
        stac_item_id=SOURCE_IMAGE['stac_item_id'],
        source_sha256=source_sha256,
        preparation_note='Single-date P3 input; pixel DN unchanged; reader offset not applied here',
    )
    temporary_path = output_path.with_name(output_path.stem + '.tmp.tif')
    with rasterio.open(temporary_path, 'w', **profile) as destination:
        destination.write(array)
        destination.write_mask(valid.astype(np.uint8) * 255)
        for index, description in enumerate(BANDS, start=1):
            destination.set_band_description(index, description)
        destination.update_tags(**output_tags)
    temporary_path.replace(output_path)

output_sha256 = sha256_file(output_path)
print('Staged:', output_path)
print('Shape:', height, width, '| valid pixels:', int(valid.sum()))
print('SHA-256:', output_sha256)
'''
            ),
            markdown("## 5. Visual QA và ghi manifest T=1"),
            code(
                r'''import matplotlib.pyplot as plt

rgb = array[[2, 1, 0]].astype(np.float32)
for channel in range(3):
    values = rgb[channel][valid]
    low, high = np.percentile(values, [2, 98])
    rgb[channel] = np.clip((rgb[channel] - low) / max(high - low, 1), 0, 1)
rgb[:, ~valid] = 0
plt.figure(figsize=(10, 8))
plt.imshow(np.moveaxis(rgb, 0, -1))
plt.title(f'{REGION_ID} single-date input {TARGET_DATE}')
plt.axis('off')
plt.show()

manifest = {
    'schema': 'prithvi-inference-dataset-v1',
    'model_input_frames': 1,
    'protocol': PROTOCOL,
    'dataset_id': DATASET_ID,
    'region_id': REGION_ID,
    'target_date': TARGET_DATE,
    'images': [{
        'path': str(output_path.relative_to(OUTPUT_ROOT)).replace('\\', '/'),
        'date': TARGET_DATE,
        'sha256': output_sha256,
    }],
    'bands': BANDS,
    'grid': {
        'crs': target_crs,
        'shape': [height, width],
        'resolution_metres': 10,
        'same_region_as_training': PROTOCOL == 'A_temporal_transfer',
    },
    'radiometry': {
        'dn_offset': -1000.0,
        'scale': 10000.0,
        'offset_verified_for_source_product': True,
        'verification_note': (
            'Sentinel_2.ipynb writes individual Planetary Computer L2A spectral values '
            'without applying the P3 reader offset; the inference reader applies -1000 once.'
        ),
    },
    'normalization': {
        'mean': [1087, 1342, 1433, 2734, 1958, 1363],
        'std': [2248, 2179, 2178, 1850, 1242, 1049],
    },
    'inference': {
        'patch_size': 224,
        'stride': 112,
        'overlap_blending': 'mean_softmax_probability',
    },
    'evaluation': {
        'date_matched_ground_truth': None,
        'status': 'qualitative until an independent date-matched label set is supplied',
    },
    'provenance': {
        'source_container': str(SOURCE_CONTAINER),
        'source_notebook': 'Sentinel_2.ipynb individual-item export',
        'stac_collection': 'sentinel-2-l2a',
        'selected_item_ids': [SOURCE_IMAGE['stac_item_id']],
        'source_filename': filename,
        'source_sha256': source_sha256,
    },
}
manifest_path = OUTPUT_ROOT / 'inference_dataset.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('Manifest:', manifest_path)
print('Manifest SHA-256:', sha256_file(manifest_path))
'''
            ),
        ],
    )


def build_inference() -> None:
    write(
        "31_inference_single_map_benchmark.ipynb",
        [
            markdown(
                """# P3 single-date inference and T4 benchmark

Notebook này chỉ load checkpoint P3 T=1, benchmark và tạo map từ một ảnh.
Nó không mở nhãn training, split, optimizer hoặc checkpoint P2 T=6."""
            ),
            markdown("## 1. Mount Drive và checkout mã nguồn"),
            code(REPO_SETUP),
            markdown("## 2. Cài môi trường và restart sạch lần đầu"),
            code(FULL_ENV),
            markdown("## 3. Khóa manifest, checkpoint P3 và output directory"),
            code(
                r'''from pathlib import Path
import hashlib

DRIVE_ROOT = Path('/content/drive/MyDrive/Prithvi_Colab')
PROTOCOL_FOLDER = 'protocol_B'
DATASET_ID = 'thai_thuy_2026_sep21_single'
MANIFEST_PATH = (
    DRIVE_ROOT / 'data' / 'inference_single' / PROTOCOL_FOLDER
    / DATASET_ID / 'inference_dataset.json'
)
RUN_ROOT = DRIVE_ROOT / 'runs' / 'P3_single_sep21_UNet_WCE_REPRO'
if not (RUN_ROOT / 'TRAINING_COMPLETE.json').is_file():
    raise FileNotFoundError('P3 training is not complete: ' + str(RUN_ROOT))
CHECKPOINT_PATH = (
    RUN_ROOT / 'best_accepted.pt'
    if (RUN_ROOT / 'best_accepted.pt').is_file()
    else RUN_ROOT / 'best_any.pt'
)
BASE_CONFIG_PATH = (
    REPO_ROOT / 'configs' / 'experiments' / 'p3_prithvi_unet_wce_single.json'
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

assert MANIFEST_PATH.is_file(), MANIFEST_PATH
assert CHECKPOINT_PATH.is_file(), CHECKPOINT_PATH
assert BASE_CONFIG_PATH.is_file(), BASE_CONFIG_PATH
CHECKPOINT_HASH_PREFIX = sha256_file(CHECKPOINT_PATH)[:12]
OUTPUT_DIR = (
    DRIVE_ROOT / 'single_inference_runs' / PROTOCOL_FOLDER
    / DATASET_ID / CHECKPOINT_HASH_PREFIX
)
print('Manifest:', MANIFEST_PATH)
print('Checkpoint:', CHECKPOINT_PATH)
print('Output:', OUTPUT_DIR)
'''
            ),
            markdown("## 4. Audit scene T=1 và strict-load checkpoint"),
            code(
                r'''import torch
from IPython.display import display
from deployment.inference_pipeline import InferenceScene, load_wce_model

if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU')
scene = InferenceScene(MANIFEST_PATH, verify=True, expected_frames=1)
model, inference_config, checkpoint_provenance, device = load_wce_model(
    BASE_CONFIG_PATH, scene.manifest, CHECKPOINT_PATH
)
assert len(scene.dates) == 1
assert inference_config['active_dates'] == scene.dates
display(scene.audit)
display(checkpoint_provenance)
print('CUDA device:', torch.cuda.get_device_name(device))
'''
            ),
            markdown("## 5. T4 benchmark"),
            code(
                r'''import json
from deployment.inference_pipeline import benchmark_inference

device_name = torch.cuda.get_device_name(device)
if 'T4' not in device_name.upper():
    raise RuntimeError(f'This benchmark is reserved for T4; active device is {device_name}')
benchmark = benchmark_inference(model, scene, device, sample_tiles=30, warmup_tiles=5)
display(benchmark)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'T4_BENCHMARK.json').write_text(
    json.dumps(benchmark, indent=2), encoding='utf-8'
)
'''
            ),
            markdown("## 6. Tạo map single-date sau khi duyệt audit và benchmark"),
            code(
                r'''from deployment.inference_pipeline import export_probability_map

RUN_MAP = False
if not RUN_MAP:
    print('Map inference is disabled. Review audit and benchmark, then set RUN_MAP=True.')
else:
    result = export_probability_map(
        model, scene, checkpoint_provenance, OUTPUT_DIR, device, progress=True
    )
    display(result)
'''
            ),
            markdown("## 7. Preview output đã hoàn tất"),
            code(
                r'''import json
import matplotlib.pyplot as plt
import rasterio

marker = OUTPUT_DIR / 'INFERENCE_COMPLETE.json'
if not marker.exists():
    print('No completed map in this output directory.')
else:
    result = json.loads(marker.read_text())
    class_file = next(name for name in result['artifacts'] if name.startswith('classes_'))
    with rasterio.open(OUTPUT_DIR / class_file) as source:
        classes = source.read(1)
    plt.figure(figsize=(12, 8))
    plt.imshow(classes, vmin=0, vmax=11, cmap='tab20')
    plt.title(f"P3 T=1 | {result['protocol']} | {result['dataset_id']} | {result['target_date']}")
    plt.axis('off')
    plt.show()
    display(result)
'''
            ),
        ],
    )


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    build_training()
    build_preparation()
    build_inference()


if __name__ == "__main__":
    main()

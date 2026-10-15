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


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[dict]) -> None:
    path = NOTEBOOKS / name
    path.write_text(
        json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )


COMMON_SETUP = r'''from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
from pathlib import Path

REPO_URL = 'https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation.git'
REPO_REF = 'main'  # Replace with an approved commit SHA before a scientific run.
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


def build_training_notebook() -> None:
    cells = [
        markdown(
            """# P2 training only: Prithvi + U-Net + WCE

Notebook này chỉ phụ trách train/resume P2 và chọn checkpoint. Nó không tạo
bản đồ, không benchmark inference và không đối chiếu điểm cũ. Notebook P2 đã
chạy đầy đủ trước đây được giữ nguyên như evidence snapshot.

Mặc định `RUN_TRAINING=False` để tránh vô tình tiêu tốn GPU. Chỉ bật sau khi
đã duyệt run directory, commit và dữ liệu."""
        ),
        markdown("## 1. Mount Drive và checkout mã nguồn"),
        code(COMMON_SETUP),
        markdown("## 2. Cài môi trường P2 theo version đã khóa"),
        code(
            r'''subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'terratorch==1.2.12', 'torchgeo==0.9.0', 'timm==1.0.30',
    'segmentation-models-pytorch==0.5.0', 'huggingface-hub==1.33.0',
    'einops==0.8.2', 'rasterio==1.5.2', 'pandas==2.2.3',
    'matplotlib>=3.8,<4', 'PyYAML>=6,<7', 'geopandas>=1,<2', 'tqdm>=4.66,<5',
    'git+https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation.git@0775770ae87b83b7d2982ed727226128685f3fba'
])

import sys
P2_SOURCE = REPO_ROOT / 'reference' / 'wce_completed'
sys.path.insert(0, str(P2_SOURCE))
print('P2 source:', P2_SOURCE)
'''
        ),
        markdown("## 3. Khóa đường dẫn dữ liệu và run training"),
        code(
            r'''from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/Prithvi_Colab')
DATA_ROOT = DRIVE_ROOT / 'data' / 'training_p2'
RUN_ID = 'P2_multitemporal6_sep21_UNet_WCE_REPRO'
RUN_ROOT = DRIVE_ROOT / 'runs' / RUN_ID
CONFIG_PATH = REPO_ROOT / 'configs' / 'experiments' / 'p2_prithvi_unet_wce_completed.json'

print('DATA_ROOT:', DATA_ROOT)
print('RUN_ROOT:', RUN_ROOT)
print('CONFIG_PATH:', CONFIG_PATH)
assert CONFIG_PATH.is_file()
'''
        ),
        markdown("## 4. Audit contract, split và dữ liệu"),
        code(
            r'''import json
from IPython.display import display
from prithvi_xtseg.training import open_experiment

experiment = open_experiment(CONFIG_PATH, DATA_ROOT, RUN_ROOT)
display(experiment.scene.audit)
assert experiment.scene.audit['shared_input_pixels'] == 0
print('Train patches:', len(experiment.manifest))
print('Validation windows:', len(experiment.val))
'''
        ),
        markdown("## 5. GPU smoke test"),
        code(
            r'''import torch

if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU')
model = experiment.load_model(pretrained=not (RUN_ROOT / 'latest_resume.pt').exists())
smoke = experiment.smoke_test()
display(smoke)
'''
        ),
        markdown("## 6. Train hoặc resume P2"),
        code(
            r'''RUN_TRAINING = False

if not RUN_TRAINING:
    print('Training is disabled. Review the paths and set RUN_TRAINING=True after approval.')
else:
    training_result = experiment.train()
    display(training_result)
'''
        ),
        markdown("## 7. Đọc model-selection artifacts, không tạo map"),
        code(
            r'''for filename in ['TRAINING_PROGRESS.json', 'TRAINING_COMPLETE.json', 'MODEL_SELECTION.json']:
    path = RUN_ROOT / filename
    if path.exists():
        print('\n', filename)
        display(json.loads(path.read_text()))

print('This notebook stops before map inference by design.')
'''
        ),
    ]
    write("10_train_p2_wce_only.ipynb", cells)


def build_data_preparation_notebook() -> None:
    cells = [
        markdown(
            """# Prepare a model-ready inference dataset

Notebook này nhận **sáu GeoTIFF riêng biệt** đã được xuất từ cell xuất từng
Sentinel-2 item của `Sentinel_2.ipynb`, kiểm tra hợp đồng đầu vào P2, chép chúng
vào dataset folder và tạo `inference_dataset.json`. Notebook không query STAC,
không train và không load checkpoint.

- Protocol A: cùng Xuân Thủy, thời gian mới.
- Protocol B: vùng mới, mùa tương đương.

Lưu ý quan trọng:

- Dùng sáu file được tạo ở bước **xuất từng item riêng** của `Sentinel_2.ipynb`.
- Không dùng file `Sentinel2_Mosaic_...tif`, vì file đó trộn pixel từ nhiều ngày.
- Sáu ngày là đầu vào đồng thời của checkpoint P2 hiện tại; notebook này chưa
  chuyển mô hình sang thiết kế single-date."""
        ),
        markdown("## 1. Mount Drive và cài thư viện kiểm tra GeoTIFF"),
        code(
            r'''from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'rasterio==1.5.2', 'pandas==2.2.3', 'matplotlib>=3.8,<4'
])
'''
        ),
        markdown("## 2. Khai báo dataset và đúng sáu file từ `Sentinel_2.ipynb`"),
        code(
            r'''from pathlib import Path

PROTOCOL = 'A_temporal_transfer'  # or 'B_spatial_transfer'
DATASET_ID = 'REPLACE_DATASET_ID'
REGION_ID = 'xuan_thuy'  # replace for Protocol B

# Có thể trỏ tới một folder đã giải nén hoặc trực tiếp tới file ZIP được tạo bởi
# Sentinel_2.ipynb. Nếu là ZIP, notebook chỉ đọc sáu filename khai báo bên dưới.
SOURCE_CONTAINER = Path(
    '/content/drive/MyDrive/Prithvi_Colab/data/sentinel2_exports/REPLACE_FOLDER_OR_ZIP'
)
PROTOCOL_FOLDER = {
    'A_temporal_transfer': 'protocol_A',
    'B_spatial_transfer': 'protocol_B',
}[PROTOCOL]
OUTPUT_ROOT = (
    Path('/content/drive/MyDrive/Prithvi_Colab/data/inference')
    / PROTOCOL_FOLDER
    / DATASET_ID
)

# Khai báo theo thứ tự thời gian tăng dần. Dùng file từng item, KHÔNG dùng mosaic.
# item_id phải là ID đầy đủ đã chọn trong Sentinel_2.ipynb, không phải phần tên
# file đã bị rút gọn.
SOURCE_IMAGES = [
    {'date': 'YYYY-MM-DD', 'filename': 'REPLACE_01.tif', 'stac_item_id': 'REPLACE_ITEM_01'},
    {'date': 'YYYY-MM-DD', 'filename': 'REPLACE_02.tif', 'stac_item_id': 'REPLACE_ITEM_02'},
    {'date': 'YYYY-MM-DD', 'filename': 'REPLACE_03.tif', 'stac_item_id': 'REPLACE_ITEM_03'},
    {'date': 'YYYY-MM-DD', 'filename': 'REPLACE_04.tif', 'stac_item_id': 'REPLACE_ITEM_04'},
    {'date': 'YYYY-MM-DD', 'filename': 'REPLACE_05.tif', 'stac_item_id': 'REPLACE_ITEM_05'},
    {'date': 'YYYY-MM-DD', 'filename': 'REPLACE_06.tif', 'stac_item_id': 'REPLACE_ITEM_06'},
]
TARGET_DATE = 'YYYY-MM-DD'  # phải bằng date của phần tử cuối

assert PROTOCOL in {'A_temporal_transfer', 'B_spatial_transfer'}
if 'REPLACE' in DATASET_ID or 'REPLACE' in str(SOURCE_CONTAINER):
    raise ValueError('Complete DATASET_ID and SOURCE_CONTAINER before continuing')
if Path(DATASET_ID).name != DATASET_ID:
    raise ValueError('DATASET_ID must be one folder name, not a path')
print(PROTOCOL, DATASET_ID, REGION_ID)
print('Source:', SOURCE_CONTAINER)
print('Output:', OUTPUT_ROOT)
'''
        ),
        markdown("## 3. Mở folder/ZIP và khóa danh sách sáu acquisition"),
        code(
            r'''import hashlib
import zipfile
from datetime import date

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

if len(SOURCE_IMAGES) != 6:
    raise ValueError('The P2 checkpoint requires exactly six source images')

filenames = [record['filename'] for record in SOURCE_IMAGES]
dates = [record['date'] for record in SOURCE_IMAGES]
item_ids = [record['stac_item_id'] for record in SOURCE_IMAGES]
if len(set(filenames)) != 6 or len(set(dates)) != 6 or len(set(item_ids)) != 6:
    raise ValueError('Filenames, dates, and full STAC item IDs must each be unique')
if any(Path(name).name != name for name in filenames):
    raise ValueError('SOURCE_IMAGES.filename must contain a basename only')
if any(Path(name).suffix.lower() not in {'.tif', '.tiff'} for name in filenames):
    raise ValueError('Every selected source file must be a GeoTIFF')
if any('mosaic' in name.lower() for name in filenames):
    raise ValueError('Do not use Sentinel2_Mosaic_*.tif; select six individual-item GeoTIFFs')
if dates != sorted(dates):
    raise ValueError('SOURCE_IMAGES must be sorted by acquisition date')
for value in dates:
    date.fromisoformat(value)
if TARGET_DATE != dates[-1]:
    raise ValueError('TARGET_DATE must equal the sixth and latest acquisition date')
if any(value.startswith('REPLACE') for value in item_ids + filenames):
    raise ValueError('Replace all filenames and full STAC item IDs')

materialized_sources = []
if SOURCE_CONTAINER.is_dir():
    for record in SOURCE_IMAGES:
        path = SOURCE_CONTAINER / record['filename']
        if not path.is_file():
            raise FileNotFoundError(path)
        materialized_sources.append(path)
elif SOURCE_CONTAINER.is_file() and zipfile.is_zipfile(SOURCE_CONTAINER):
    staging = Path('/content') / f'sentinel2_selected_{DATASET_ID}'
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_CONTAINER) as archive:
        file_members = [name for name in archive.namelist() if not name.endswith('/')]
        by_basename = {}
        for member in file_members:
            by_basename.setdefault(Path(member).name, []).append(member)
        for record in SOURCE_IMAGES:
            matches = by_basename.get(record['filename'], [])
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one ZIP member named {record['filename']}, found {len(matches)}"
                )
            destination = staging / record['filename']
            with archive.open(matches[0]) as source, open(destination, 'wb') as target:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(block)
            materialized_sources.append(destination)
else:
    raise FileNotFoundError('SOURCE_CONTAINER must be a folder or a ZIP file')

print('Locked input dates:', dates)
for path in materialized_sources:
    print(path.name, sha256_file(path)[:12])
'''
        ),
        markdown("## 4. Audit band/date/grid và stage sáu GeoTIFF"),
        code(
            r'''import json
import numpy as np
import pandas as pd
import rasterio

BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
rasters_dir = OUTPUT_ROOT / 'rasters'
rasters_dir.mkdir(parents=True, exist_ok=True)

image_records = []
audit_rows = []
valid_masks = []
reference_grid = None

for record, source_path in zip(SOURCE_IMAGES, materialized_sources):
    acquisition_date = record['date']
    output_path = rasters_dir / f'Sentinel2_{acquisition_date}_{REGION_ID}_6bands_10m.tif'

    source_sha256 = sha256_file(source_path)
    with rasterio.open(source_path) as source:
        if source.count != 6 or list(source.descriptions) != BANDS:
            raise ValueError(f'Wrong band contract: {source_path.name}')
        if source.tags().get('acquisition_date') != acquisition_date:
            raise ValueError(f'acquisition_date tag differs: {source_path.name}')
        if source.tags().get('source') != 'Microsoft Planetary Computer - Sentinel-2 L2A':
            raise ValueError(f'Unexpected source tag: {source_path.name}')
        if source.crs is None or not source.crs.is_projected:
            raise ValueError(f'GeoTIFF must use a projected CRS: {source_path.name}')
        if not (
            np.isclose(abs(source.transform.a), 10.0)
            and np.isclose(abs(source.transform.e), 10.0)
            and np.isclose(source.transform.b, 0.0)
            and np.isclose(source.transform.d, 0.0)
        ):
            raise ValueError(f'GeoTIFF is not on a north-up 10 m grid: {source_path.name}')
        if list(source.dtypes) != ['uint16'] * 6 or source.nodata != 0:
            raise ValueError(f'Expected uint16 raw DN with nodata=0: {source_path.name}')

        grid = (str(source.crs), source.transform, source.width, source.height)
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError(f'Raster grid differs from the first acquisition: {source_path.name}')

        array = source.read()
        valid = (source.read_masks() > 0).all(0) & np.isfinite(array).all(0)
        if not valid.any():
            raise ValueError(f'No valid pixel in {source_path.name}')
        array[:, ~valid] = 0
        profile = source.profile.copy()
        source_tags = source.tags()

    if output_path.exists():
        with rasterio.open(output_path) as existing:
            if existing.tags().get('source_sha256') != source_sha256:
                raise FileExistsError(
                    f'{output_path} exists for different source bytes; use a new DATASET_ID'
                )
        print('Reusing staged file:', output_path.name)
    else:
        profile.update(compress='deflate', tiled=True, nodata=0)
        output_tags = dict(source_tags)
        output_tags.update(
            acquisition_date=acquisition_date,
            stac_item_id=record['stac_item_id'],
            source_sha256=source_sha256,
            preparation_note=(
                'Staged from Sentinel_2.ipynb individual-item export; pixel DN unchanged; '
                'reader offset is not applied here'
            ),
        )
        temporary_path = output_path.with_name(output_path.stem + '.tmp.tif')
        with rasterio.open(temporary_path, 'w', **profile) as destination:
            destination.write(array)
            destination.write_mask(valid.astype(np.uint8) * 255)
            for index, description in enumerate(BANDS, start=1):
                destination.set_band_description(index, description)
            destination.update_tags(**output_tags)
        temporary_path.replace(output_path)

    valid_masks.append(valid)
    output_sha256 = sha256_file(output_path)
    image_records.append({
        'path': str(output_path.relative_to(OUTPUT_ROOT)).replace('\\', '/'),
        'date': acquisition_date,
        'sha256': output_sha256,
        'stac_item_id': record['stac_item_id'],
        'source_filename': record['filename'],
        'source_sha256': source_sha256,
    })
    audit_rows.append({
        'date': acquisition_date,
        'source_file': record['filename'],
        'valid_pixels': int(valid.sum()),
        'source_sha256_12': source_sha256[:12],
        'output_sha256_12': output_sha256[:12],
    })

TARGET_CRS, transform, width, height = reference_grid
TARGET_RESOLUTION = 10
common_valid = np.logical_and.reduce(valid_masks)
if not common_valid.any():
    raise ValueError('No pixel is valid in all six acquisitions')

display(pd.DataFrame(audit_rows))
print('Grid:', TARGET_CRS, height, width, transform)
print('Common valid pixels:', int(common_valid.sum()))
'''
        ),
        markdown("## 5. Visual QA cho target-date raster và common-valid mask"),
        code(
            r'''import matplotlib.pyplot as plt

target_path = OUTPUT_ROOT / image_records[-1]['path']
with rasterio.open(target_path) as source:
    rgb = source.read([3, 2, 1]).astype(np.float32)
    mask = source.dataset_mask() > 0
for channel in range(3):
    values = rgb[channel][mask]
    low, high = np.percentile(values, [2, 98])
    rgb[channel] = np.clip((rgb[channel] - low) / max(high - low, 1), 0, 1)
rgb[:, ~mask] = 0
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(np.moveaxis(rgb, 0, -1))
axes[0].set_title(f'{REGION_ID} target image {TARGET_DATE}')
axes[1].imshow(common_valid, cmap='gray')
axes[1].set_title('Pixels valid in all six acquisitions')
for axis in axes:
    axis.axis('off')
plt.tight_layout()
plt.show()
'''
        ),
        markdown("## 6. Ghi inference manifest với provenance từ `Sentinel_2.ipynb`"),
        code(
            r'''# Sentinel_2.ipynb individual-item export reads the Planetary Computer L2A
# spectral assets, resamples to the common grid, and writes uint16 DN without
# subtracting 1000. Notebook 30/inference_pipeline remains the only place that
# applies the locked reader offset and normalization. This confirms that the
# local export path does not apply the offset twice; it does not independently
# prove the upstream Planetary Computer radiometric convention.
OFFSET_VERIFICATION_NOTE = (
    'Locked P2 radiometry decision. Sentinel_2.ipynb writes Microsoft Planetary '
    'Computer Sentinel-2 L2A spectral COG values as uint16 without applying an '
    'offset or normalization, so the reader remains the only -1000 application. '
    'The upstream source-product convention remains an explicit project assumption.'
)

manifest = {
    'schema': 'prithvi-inference-dataset-v1',
    'protocol': PROTOCOL,
    'dataset_id': DATASET_ID,
    'region_id': REGION_ID,
    'target_date': TARGET_DATE,
    'images': [{key: record[key] for key in ['path', 'date', 'sha256']} for record in image_records],
    'bands': BANDS,
    'grid': {
        'crs': TARGET_CRS,
        'shape': [height, width],
        'resolution_metres': TARGET_RESOLUTION,
        'same_region_as_training': PROTOCOL == 'A_temporal_transfer',
    },
    'radiometry': {
        'dn_offset': -1000.0,
        'scale': 10000.0,
        'offset_verified_for_source_product': True,
        'verification_note': OFFSET_VERIFICATION_NOTE,
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
        'source_notebook': 'Sentinel_2.ipynb individual-item export (mosaic excluded)',
        'stac_collection': 'sentinel-2-l2a',
        'selected_item_ids': item_ids,
        'source_images': [
            {
                'date': record['date'],
                'filename': record['source_filename'],
                'sha256': record['source_sha256'],
                'stac_item_id': record['stac_item_id'],
            }
            for record in image_records
        ],
    },
}
manifest_path = OUTPUT_ROOT / 'inference_dataset.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('Manifest:', manifest_path)
print('Manifest SHA-256:', sha256_file(manifest_path))
'''
        ),
    ]
    write("20_prepare_inference_dataset.ipynb", cells)


def build_inference_notebook() -> None:
    cells = [
        markdown(
            """# P2 inference map and T4 benchmark

Notebook này chỉ load checkpoint đã chọn và dataset manifest của Protocol A
hoặc B. Nó không train, không resume optimizer và không đọc label raster.

Benchmark và map output dùng thư mục riêng, được khóa bằng SHA-256 của manifest
và checkpoint."""
        ),
        markdown("## 1. Mount Drive và checkout mã nguồn"),
        code(COMMON_SETUP),
        markdown("## 2. Cài inference environment"),
        code(
            r'''subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'terratorch==1.2.12', 'torchgeo==0.9.0', 'timm==1.0.30',
    'segmentation-models-pytorch==0.5.0', 'huggingface-hub==1.33.0',
    'einops==0.8.2', 'rasterio==1.5.2', 'pandas==2.2.3',
    'matplotlib>=3.8,<4', 'tqdm>=4.66,<5',
    'git+https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation.git@0775770ae87b83b7d2982ed727226128685f3fba'
])

import sys
sys.path.insert(0, str(REPO_ROOT / 'reference' / 'wce_completed'))
sys.path.insert(0, str(REPO_ROOT))
'''
        ),
        markdown("## 3. Chọn dataset manifest, checkpoint và output riêng"),
        code(
            r'''from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/Prithvi_Colab')
MANIFEST_PATH = DRIVE_ROOT / 'data' / 'inference' / 'REPLACE' / 'inference_dataset.json'
CHECKPOINT_PATH = DRIVE_ROOT / 'runs' / 'P2_multitemporal6_sep21_UNet_WCE' / 'best_any.pt'
BASE_CONFIG_PATH = REPO_ROOT / 'configs' / 'experiments' / 'p2_prithvi_unet_wce_completed.json'

# Use a new directory for every dataset/checkpoint pair. The inference code seals it.
OUTPUT_DIR = DRIVE_ROOT / 'inference_runs' / 'REPLACE_PROTOCOL' / 'REPLACE_DATASET' / 'REPLACE_CHECKPOINT_HASH_PREFIX'

assert MANIFEST_PATH.is_file(), MANIFEST_PATH
assert CHECKPOINT_PATH.is_file(), CHECKPOINT_PATH
assert BASE_CONFIG_PATH.is_file(), BASE_CONFIG_PATH
print(MANIFEST_PATH, CHECKPOINT_PATH, OUTPUT_DIR, sep='\n')
'''
        ),
        markdown("## 4. Audit dataset và load checkpoint P2"),
        code(
            r'''import torch
from IPython.display import display
from deployment.inference_pipeline import InferenceScene, load_wce_model

if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU')

scene = InferenceScene(MANIFEST_PATH, verify=True)
model, inference_config, checkpoint_provenance, device = load_wce_model(
    BASE_CONFIG_PATH, scene.manifest, CHECKPOINT_PATH
)
display(scene.audit)
display(checkpoint_provenance)
print('CUDA device:', torch.cuda.get_device_name(device))
'''
        ),
        markdown("## 5. T4 benchmark cell"),
        code(
            r'''import json
from deployment.inference_pipeline import benchmark_inference

REQUIRE_T4 = True
device_name = torch.cuda.get_device_name(device)
if REQUIRE_T4 and 'T4' not in device_name.upper():
    raise RuntimeError(f'This benchmark is reserved for a T4 runtime; active device is {device_name}')

benchmark = benchmark_inference(model, scene, device, sample_tiles=30, warmup_tiles=5)
display(benchmark)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
benchmark_path = OUTPUT_DIR / 'T4_BENCHMARK.json'
benchmark_path.write_text(json.dumps(benchmark, indent=2), encoding='utf-8')
print('Saved:', benchmark_path)
'''
        ),
        markdown(
            """## 6. Tạo map inference

`RUN_MAP` mặc định tắt. Khi bật, output được ghi cùng class, confidence,
entropy, common-validity, lock và measured full-map wall time. Không diễn giải
F1/IoU nếu chưa có ground truth độc lập đúng thời gian."""
        ),
        code(
            r'''from deployment.inference_pipeline import export_probability_map

RUN_MAP = False
if not RUN_MAP:
    print('Map inference is disabled. Review the audit and benchmark before setting RUN_MAP=True.')
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
    plt.title(f"{result['protocol']} | {result['dataset_id']} | {result['target_date']}")
    plt.axis('off')
    plt.show()
    display(result)
'''
        ),
    ]
    write("30_inference_map_benchmark.ipynb", cells)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    build_training_notebook()
    build_data_preparation_notebook()
    build_inference_notebook()


if __name__ == "__main__":
    main()

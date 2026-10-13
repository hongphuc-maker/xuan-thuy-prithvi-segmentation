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
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


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

Notebook này tạo dữ liệu cho **Protocol A** hoặc **Protocol B**. Đầu ra là sáu
GeoTIFF đã đồng đăng ký và `inference_dataset.json`. Notebook không train và
không load checkpoint.

- Protocol A: cùng Xuân Thủy, thời gian mới.
- Protocol B: vùng mới, mùa tương đương.

Phải chọn chính xác sáu STAC item ID. Không tự động lấy ảnh có cloud cover thấp
nhất rồi xem đó là quyết định khoa học."""
        ),
        markdown("## 1. Mount Drive và cài thư viện chuẩn bị dữ liệu"),
        code(
            r'''from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'pystac-client', 'planetary-computer', 'geopandas>=1,<2',
    'rasterio==1.5.2', 'shapely', 'pyproj', 'pandas==2.2.3', 'matplotlib>=3.8,<4'
])
'''
        ),
        markdown("## 2. Khai báo Protocol A hoặc B và AOI"),
        code(
            r'''from pathlib import Path

PROTOCOL = 'A_temporal_transfer'  # or 'B_spatial_transfer'
DATASET_ID = 'REPLACE_DATASET_ID'
REGION_ID = 'xuan_thuy'  # replace for Protocol B
AOI_PATH = Path('/content/drive/MyDrive/Prithvi_Colab/data/inference/REPLACE/aoi/aoi.geojson')
OUTPUT_ROOT = Path('/content/drive/MyDrive/Prithvi_Colab/data/inference/REPLACE')

SEARCH_START = 'YYYY-MM-DD'
SEARCH_END = 'YYYY-MM-DD'
TARGET_CRS = 'EPSG:32648'  # Confirm the correct projected CRS for the AOI.
TARGET_RESOLUTION = 10
MAX_SCENE_CLOUD = 40

assert PROTOCOL in {'A_temporal_transfer', 'B_spatial_transfer'}
assert AOI_PATH.is_file(), AOI_PATH
print(PROTOCOL, DATASET_ID, REGION_ID, OUTPUT_ROOT)
'''
        ),
        markdown("## 3. Query candidates và duyệt metadata"),
        code(
            r'''import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from pystac_client import Client
import planetary_computer

aoi = gpd.read_file(AOI_PATH)
if aoi.crs is None:
    raise ValueError('AOI has no CRS')
aoi_4326 = aoi.to_crs('EPSG:4326')
search_geometry = mapping(aoi_4326.geometry.union_all())

catalog = Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=planetary_computer.sign_inplace,
)
search = catalog.search(
    collections=['sentinel-2-l2a'],
    intersects=search_geometry,
    datetime=f'{SEARCH_START}/{SEARCH_END}',
    query={'eo:cloud_cover': {'lt': MAX_SCENE_CLOUD}},
)
items = list(search.items())
candidates = pd.DataFrame([
    {
        'item_id': item.id,
        'date': item.datetime.date().isoformat(),
        'cloud_cover_scene_percent': item.properties.get('eo:cloud_cover'),
        'processing_baseline': item.properties.get('s2:processing_baseline'),
        'platform': item.properties.get('platform'),
        'tile': item.properties.get('s2:mgrs_tile'),
    }
    for item in items
]).sort_values(['date', 'cloud_cover_scene_percent'])
display(candidates)
print('Candidate count:', len(candidates))
'''
        ),
        markdown("## 4. Chốt sáu item ID và quyết định offset"),
        code(
            r'''SELECTED_ITEM_IDS = [
    'REPLACE_ITEM_01',
    'REPLACE_ITEM_02',
    'REPLACE_ITEM_03',
    'REPLACE_ITEM_04',
    'REPLACE_ITEM_05',
    'REPLACE_ITEM_06',
]
TARGET_DATE = 'YYYY-MM-DD'

# Set True only after reviewing processing_baseline and confirming that the
# exported rasters retain the raw DN convention expected by the -1000 reader.
OFFSET_VERIFIED_FOR_SOURCE_PRODUCT = False
OFFSET_VERIFICATION_NOTE = 'REPLACE with the metadata/export evidence used for this decision'

if len(SELECTED_ITEM_IDS) != 6 or len(set(SELECTED_ITEM_IDS)) != 6:
    raise ValueError('Select exactly six unique STAC item IDs')
selected = [item for item in items if item.id in SELECTED_ITEM_IDS]
if len(selected) != 6:
    missing = sorted(set(SELECTED_ITEM_IDS) - {item.id for item in selected})
    raise ValueError(f'Selected items were not returned by the query: {missing}')
selected = sorted(selected, key=lambda item: item.datetime)
selected_dates = [item.datetime.date().isoformat() for item in selected]
if selected_dates[-1] != TARGET_DATE:
    raise ValueError('TARGET_DATE must equal the final selected acquisition date')
display(candidates[candidates.item_id.isin(SELECTED_ITEM_IDS)].sort_values('date'))
print('Selected dates:', selected_dates)
'''
        ),
        markdown("## 5. Tạo common 10 m grid và xuất sáu GeoTIFF"),
        code(
            r'''import hashlib
import json
import math
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
rasters_dir = OUTPUT_ROOT / 'rasters'
rasters_dir.mkdir(parents=True, exist_ok=True)

aoi_projected = aoi.to_crs(TARGET_CRS)
aoi_geometry = aoi_projected.geometry.union_all()
minx, miny, maxx, maxy = aoi_geometry.bounds
left = math.floor(minx / TARGET_RESOLUTION) * TARGET_RESOLUTION
bottom = math.floor(miny / TARGET_RESOLUTION) * TARGET_RESOLUTION
right = math.ceil(maxx / TARGET_RESOLUTION) * TARGET_RESOLUTION
top = math.ceil(maxy / TARGET_RESOLUTION) * TARGET_RESOLUTION
width = int(round((right - left) / TARGET_RESOLUTION))
height = int(round((top - bottom) / TARGET_RESOLUTION))
transform = from_origin(left, top, TARGET_RESOLUTION, TARGET_RESOLUTION)
inside_aoi = geometry_mask(
    [aoi_geometry.__geo_interface__],
    out_shape=(height, width),
    transform=transform,
    invert=True,
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

image_records = []
for item in selected:
    acquisition_date = item.datetime.date().isoformat()
    output_path = rasters_dir / f'Sentinel2_{acquisition_date}_{REGION_ID}_6bands_10m.tif'
    stack = np.zeros((6, height, width), dtype=np.uint16)
    valid = inside_aoi.copy()
    for band_index, band_name in enumerate(BANDS):
        if band_name not in item.assets:
            raise KeyError(f'{band_name} is missing from STAC item {item.id}')
        with rasterio.open(item.assets[band_name].href) as source:
            with WarpedVRT(
                source,
                crs=TARGET_CRS,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                nodata=0,
            ) as vrt:
                band = vrt.read(1, masked=True, out_dtype='float32')
                band_valid = ~np.ma.getmaskarray(band) & np.isfinite(band.filled(np.nan))
                valid &= band_valid
                values = np.rint(band.filled(0)).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
                stack[band_index] = values
    stack[:, ~valid] = 0
    profile = {
        'driver': 'GTiff', 'height': height, 'width': width, 'count': 6,
        'dtype': 'uint16', 'crs': TARGET_CRS, 'transform': transform,
        'nodata': 0, 'compress': 'deflate', 'tiled': True,
    }
    with rasterio.open(output_path, 'w', **profile) as destination:
        destination.write(stack)
        destination.write_mask(valid.astype(np.uint8) * 255)
        for index, description in enumerate(BANDS, start=1):
            destination.set_band_description(index, description)
        destination.update_tags(
            acquisition_date=acquisition_date,
            stac_item_id=item.id,
            processing_baseline=str(item.properties.get('s2:processing_baseline', 'unknown')),
            export_note='Raw source DN reprojected to common 10 m grid; reader offset is not applied here',
        )
    image_records.append({
        'path': str(output_path.relative_to(OUTPUT_ROOT)).replace('\\', '/'),
        'date': acquisition_date,
        'sha256': sha256_file(output_path),
        'stac_item_id': item.id,
    })
    print(output_path.name, int(valid.sum()))

print('Grid:', TARGET_CRS, height, width, transform)
'''
        ),
        markdown("## 6. Visual QA cho target-date raster"),
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
plt.figure(figsize=(10, 8))
plt.imshow(np.moveaxis(rgb, 0, -1))
plt.title(f'{REGION_ID} target image {TARGET_DATE}')
plt.axis('off')
plt.show()
'''
        ),
        markdown("## 7. Ghi inference manifest sau khi offset được xác nhận"),
        code(
            r'''if not OFFSET_VERIFIED_FOR_SOURCE_PRODUCT:
    raise RuntimeError(
        'Stop: review source processing_baseline and export radiometry, then set '
        'OFFSET_VERIFIED_FOR_SOURCE_PRODUCT=True with a verification note.'
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
        'aoi_source': str(AOI_PATH),
        'stac_collection': 'sentinel-2-l2a',
        'selected_item_ids': SELECTED_ITEM_IDS,
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

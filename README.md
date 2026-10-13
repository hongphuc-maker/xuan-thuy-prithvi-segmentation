# Xuan Thuy Prithvi Segmentation

Research code for 11-class semantic segmentation at Xuan Thuy using the
Prithvi-EO-2.0-300M-TL foundation model, a UNet decoder, and six Sentinel-2
observations. The target map date is 21 September 2026.

This repository separates completed evidence from new transfer workflows:

| Experiment | Initialization and objective | Status |
| --- | --- | --- |
| P2 | Prithvi pretrained encoder + UNet decoder + weighted cross-entropy | Completed and evaluated |

## Completed WCE result

The selected checkpoint occurred at optimizer step 2,184:

| Validation measure | Value |
| --- | ---: |
| Macro-F1 over 11 fixed classes | 0.442924 |
| Macro-IoU over 11 fixed classes | 0.356342 |
| Overall agreement with the historical validation raster | 0.838300 |
| Predicted classes | 10 of 11 |

The acceptance gate did not pass, so the 21 September map is an exploratory
output. Agreement on 1,036 usable previously used reference points was 0.921815,
with macro-F1 0.903198 over the seven represented classes. These points have
unverified date alignment and do not establish independent September accuracy.
See [`results/wce_completed_2026-10-04.json`](results/wce_completed_2026-10-04.json)
and [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).

## Repository scope

The public repository contains source code, immutable experiment contracts,
tests, and Colab notebooks. It intentionally excludes:

- Sentinel-2 and label rasters;
- reference-point archives;
- pretrained and trained checkpoints;
- prediction GeoTIFFs and run directories;
- Google Drive contents or credentials.

Expected file names and SHA-256 checksums are recorded in the experiment JSON
files. Dataset access remains separate from this source-code license.

```text
configs/       immutable P2 contract and Protocol A/B inference templates
docs/          data, status, and reproducibility notes
notebooks/     evidence notebooks plus separated train, data-prep, and inference notebooks
deployment/    label-free P2 inference, map export, and benchmark helpers
reference/     frozen source snapshot used by the completed WCE notebook
results/       machine-readable summaries of completed evidence
tests/         lightweight P2 and inference-contract tests
```

## Run on Colab

The completed notebook is retained as an executed evidence snapshot:

[Open the completed WCE notebook in Colab](https://colab.research.google.com/github/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/main/notebooks/01_prithvi_unet_wce_completed.ipynb)

Place the private data files under the notebook's `DATA_ROOT`, verify every
checksum, select a GPU runtime, and run from the first cell.

### Standardized execution notebooks

New Protocol A and Protocol B work is separated by responsibility:

1. [`10_train_p2_wce_only.ipynb`](notebooks/10_train_p2_wce_only.ipynb)
   trains or resumes P2 and stops after checkpoint selection;
2. [`20_prepare_inference_dataset.ipynb`](notebooks/20_prepare_inference_dataset.ipynb)
   prepares six aligned acquisitions and writes a locked inference manifest;
3. [`30_inference_map_benchmark.ipynb`](notebooks/30_inference_map_benchmark.ipynb)
   loads one selected P2 checkpoint, benchmarks T4 inference, and optionally
   writes a label-free map.

The inference path does not load the training label, split, optimizer, or WCE
loss. See
[`docs/STANDARDIZED_WORKFLOW_PROTOCOLS_A_B.md`](docs/STANDARDIZED_WORKFLOW_PROTOCOLS_A_B.md)
for the Drive layout, approval gates, and interpretation boundaries. Protocol C
is deferred.

## Local development

Python 3.10 or 3.11 is recommended for lightweight tests:

```bash
git clone https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation.git
cd xuan-thuy-prithvi-segmentation
python -m pip install -e ".[test]"
pytest
```

The full training stack is designed for a CUDA-enabled Colab runtime:

```bash
python -m pip install -e ".[train,test]"
```

The code reuses data-contract and split helpers from the public
[`xuan-thuy-dmi-segmentation`](https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation)
repository at the pinned baseline commit recorded in each experiment contract.

## Scientific boundaries

- Six bands are used in this order: B02, B03, B04, B8A, B11, and B12.
- The reader applies DN offset -1000 and the locked normalization statistics.
- No SCL/cloud masks are supplied; nodata validity is not cloud validity.
- The label raster is historical and has no confirmed September reference date.
- Protocol A and Protocol B have prepared workflows but no reported transfer
  result until their locked inference artifacts exist.

## License

Source code is released under the [BSD 3-Clause License](LICENSE). Data,
labels, reference points, pretrained weights, checkpoints, and model outputs
are separate research assets and are not licensed by this repository.

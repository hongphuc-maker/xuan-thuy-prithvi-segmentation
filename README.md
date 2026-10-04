# Xuan Thuy Prithvi Segmentation

Research code for 11-class semantic segmentation at Xuan Thuy using the
Prithvi-EO-2.0-300M-TL foundation model, a UNet decoder, and six Sentinel-2
observations. The target map date is 21 September 2026.

This repository separates completed evidence from planned work:

| Branch | Initialization and objective | Status |
| --- | --- | --- |
| P2 | Prithvi pretrained encoder + UNet decoder + weighted cross-entropy | Completed and evaluated |
| P5 | Frozen Prithvi encoder + WCE task-head warm-up, then exact DMI + smooth logit morphology | Notebook and source prepared; training not started |

The P5 branch does not load the completed WCE checkpoint. It starts from the
original Prithvi pretrained encoder, keeps the encoder frozen, and warms up the
randomly initialized task-specific neck, decoder, and classification head.
Transition to DMI is allowed only after the numerical and class-coverage gate
passes twice.

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
configs/       immutable P2 and P5 experiment contracts
docs/          data, status, and reproducibility notes
notebooks/     Sentinel-2 export, completed WCE run, and prepared P5 run
reference/     frozen source snapshot used by the completed WCE notebook
results/       machine-readable summaries of completed evidence
src/           current P5 Prithvi/DMI/morphology implementation
tests/         lightweight contract and numerical tests
```

## Run on Colab

The completed notebook is retained as an executed evidence snapshot:

[Open the completed WCE notebook in Colab](https://colab.research.google.com/github/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/main/notebooks/01_prithvi_unet_wce_completed.ipynb)

The next experiment is prepared but has no results yet:

[Open the staged WCE warm-up to DMI + morphology notebook in Colab](https://colab.research.google.com/github/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/main/notebooks/02_frozen_prithvi_wce_warmup_dmi_morphology.ipynb)

Place the private data files under the notebook's `DATA_ROOT`, verify every
checksum, select a GPU runtime, and run from the first cell. Do not treat the
P5 notebook as completed until its own run directory contains the selection,
map, and evaluation artifacts.

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
- P5 is a planned experiment. No DMI or morphology performance result is claimed.

## License

Source code is released under the [BSD 3-Clause License](LICENSE). Data,
labels, reference points, pretrained weights, checkpoints, and model outputs
are separate research assets and are not licensed by this repository.

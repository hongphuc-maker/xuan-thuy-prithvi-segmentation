# Standardized workflow for Protocol A and Protocol B

Status: design and executable notebooks prepared. No Protocol A or Protocol B
inference result is claimed until a dataset manifest and run artifacts exist.
Protocol C is intentionally out of scope for this phase.

## 1. Why the workflow is separated

The completed P2 notebook combined model training, checkpoint selection, full
scene mapping, and historical-point comparison. That notebook remains an
evidence snapshot. New work uses three separate executable notebooks so that a
map run cannot silently retrain the model or reuse a training label.

| Notebook | Responsibility | Inputs | Outputs |
| --- | --- | --- | --- |
| `10_train_p2_wce_only.ipynb` | Train or resume P2 and select a checkpoint | Locked P2 config, six training images, historical label | Checkpoints, histories, model selection |
| `20_prepare_inference_dataset.ipynb` | Audit and stage a six-date stack for a target region and time | Folder or ZIP from the individual-item export in `Sentinel_2.ipynb`, six exact filenames and STAC item IDs | Six audited GeoTIFFs and `inference_dataset.json` |
| `30_inference_map_benchmark.ipynb` | Load one selected P2 checkpoint, benchmark, and create a map | Checkpoint, P2 config, inference manifest | Benchmark JSON, map GeoTIFFs, inference lock |

The inference notebooks do not open the training label raster, training split,
optimizer state, or WCE loss.

The default inference path points to the already selected checkpoint in
`P2_multitemporal6_sep21_UNet_WCE`. A checkpoint from a reproduction run must
not replace it silently; approve the new checkpoint hash and edit the inference
path explicitly.

## 2. Drive layout

Use separate roots for training and inference. Never point a Protocol A or B
output at the completed P2 training run.

```text
MyDrive/Prithvi_Colab/
├── data/
│   ├── training_p2/
│   └── inference/
│       ├── protocol_A/<dataset_id>/
│       │   ├── aoi/
│       │   ├── rasters/
│       │   └── inference_dataset.json
│       └── protocol_B/<dataset_id>/
│           ├── aoi/
│           ├── rasters/
│           └── inference_dataset.json
├── runs/
│   ├── P2_multitemporal6_sep21_UNet_WCE/
│   │   └── best_any.pt
│   └── P2_multitemporal6_sep21_UNet_WCE_REPRO/
│       └── training and checkpoint-selection artifacts from notebook 10
└── inference_runs/
    ├── protocol_A/<dataset_id>/<checkpoint_hash_prefix>/
    └── protocol_B/<dataset_id>/<checkpoint_hash_prefix>/
```

## 3. Locked model-facing input contract

The current P2 checkpoint expects the same tensor structure used during
training:

- exactly six acquisitions, sorted by date;
- the target date is the last acquisition;
- bands in the order `B02, B03, B04, B8A, B11, B12`;
- a common projected 10 m grid for all acquisitions;
- reader offset `-1000`, followed by the locked P2 normalization;
- 224 by 224 inference tiles with stride 112;
- mean blending of softmax probabilities in overlap regions.

`Sentinel_2.ipynb` writes the selected individual-item spectral COG values as
uint16 DN on a common grid without applying the reader offset or normalization.
The data-preparation notebook audits that export and stages the pixels without
changing valid DN values. The inference reader applies the offset. Do not use
the annual mosaic export, because that file can mix observations from different
dates within one raster. This verifies that the local export path does not apply
the offset twice; the upstream Planetary Computer radiometric convention remains
the same explicit project assumption as in the completed P2 contract.

## 4. Protocol A: temporal transfer

Question: can the frozen selected P2 checkpoint map Xuân Thủy at another time?

### Fixed elements

- geographic area: Xuân Thủy;
- selected P2 checkpoint and architecture;
- six-band order, preprocessing, patch size, stride, and probability blending;
- no fine-tuning on the new period;
- no checkpoint choice based on the new-period result.

### Changed element

- a new sequence of six acquisitions ending at a new target date.

The six acquisition dates should follow a comparable seasonal logic to the P2
sequence. If the temporal spacing changes substantially, record that as an
additional domain shift rather than attributing the effect only to the target
date.

### Evidence levels

- Without a date-matched label set: qualitative temporal-transfer inference.
- With an independent date-matched label set: quantitative temporal-transfer
  evaluation. The label set must not influence checkpoint selection or data
  preprocessing decisions.

## 5. Protocol B: spatial transfer

Question: can the same frozen selected P2 checkpoint map a new area during a
comparable season?

### Fixed elements

- selected P2 checkpoint and architecture;
- six-frame and six-band model contract;
- preprocessing, patch size, stride, and probability blending;
- no fine-tuning on the new region;
- target season chosen to be comparable with the original January-to-September
  sequence.

### Changed element

- the AOI and its latitude/longitude coordinates;
- the six Sentinel-2 acquisitions covering that AOI.

Record land-cover composition, cloud conditions, sensor/product metadata, and
AOI size. A new-area result without local ground truth demonstrates that the
pipeline runs, but does not establish accuracy.

## 6. Data preparation acceptance gate

Do not start inference until all checks pass:

1. six exact individual-item filenames and full STAC item IDs are recorded;
2. no `Sentinel2_Mosaic_*.tif` file is selected;
3. acquisition dates are unique, sorted, and end at the target date;
4. every GeoTIFF has six bands with the required descriptions and the expected
   Planetary Computer source tag;
5. all files are uint16 raw DN with nodata 0 and share the same projected,
   north-up 10 m CRS, transform, width, and height;
6. raster masks identify valid pixels and the six-date intersection is not empty;
7. the radiometric offset decision is documented from the export code path;
8. source and staged-file SHA-256 values are stored in the manifest;
9. the manifest identifies Protocol A or Protocol B;
10. the dataset folder is read-only after the manifest is approved.

## 7. Inference and benchmark artifacts

The inference notebook writes:

- `INFERENCE_LOCK.json`, binding the dataset manifest, checkpoint, patch size,
  stride, and blending rule;
- `T4_BENCHMARK.json`, containing measured batch-1 tile latency;
- a class GeoTIFF for the target date;
- uncalibrated confidence and normalized entropy GeoTIFFs;
- `common_validity.tif`;
- `INFERENCE_COMPLETE.json` with hashes and measured full-map wall time.

The benchmark reports two timings:

- compute latency for model forward plus softmax;
- end-to-end tile latency including crop normalization, host-to-device transfer,
  model execution, and synchronization.

The estimated full-map time is only an estimate derived from sampled tiles.
The measured full-map wall time in `INFERENCE_COMPLETE.json` is the operational
number to report after map generation finishes.

## 8. Evaluation and leakage boundary

The completed P2 validation is an internal spatial holdout within the original
six acquisitions. The completed full-scene map also contains pixels from the
training region, so the full map is a fitted product map rather than an
independent test map.

Protocol A and B use inference-only folders and a frozen checkpoint. If a
date-matched label becomes available, evaluation must happen after the map is
sealed. Do not use that label to choose dates, cloud filtering, checkpoint,
stride, or post-processing.

## 9. Why a map can look smooth

The current workflow does not create six class masks and average them. It feeds
six dates jointly to the encoder, reads the target-date representation after
temporal interaction, and predicts one probability cube for each tile.

Several mechanisms can affect visual smoothness:

- 16 by 16 patch tokenization and transformer context;
- learned pyramid construction and U-Net decoding;
- 224 by 224 contextual tiles;
- 50 percent tile overlap with probability averaging;
- the spatial continuity of the landscape and labels.

Visual smoothness alone does not identify which mechanism dominates. A later
matched ablation should hold the checkpoint and data fixed, then compare
overlap averaging with non-overlapping inference and compare multitemporal input
with a separately trained single-date control. That ablation is not part of
Protocol A or B and will require separate approval before compute is used.

## 10. Approval points before any run

The team should approve these items for each dataset:

- protocol and dataset ID;
- AOI and target date;
- six selected STAC item IDs;
- radiometric offset decision;
- selected checkpoint path and SHA-256;
- whether the run is qualitative or has independent date-matched ground truth;
- expected Colab runtime and storage location.

Preparing code and manifests does not authorize training, inference, or changes
to the locked P2 experimental design.

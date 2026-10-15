# P3 single-date branch

Status: executable code and notebooks prepared. No P3 accuracy or transfer
result is claimed until Colab training and artifact readback are complete.

## Objective

P3 answers whether a model fine-tuned from the same Prithvi foundation encoder
can operate on one Sentinel-2 acquisition instead of the six-date P2 sequence.
It is also the matched control needed to assess whether temporal input improves
or harms transfer.

P3 is not P2 with five frames removed. It is a separate downstream model:

```text
[B, 6 bands, T=1, 224, 224]
  -> Prithvi-EO-2.0-300M-TL encoder configured for one frame
  -> single target-date feature slice
  -> learned spatial pyramid
  -> U-Net decoder [512, 256, 128, 64]
  -> 11-class head
```

## Fixed comparison contract

P3 retains the P2 label raster, spatial split, 48-pixel guard, patch manifest,
batch schedule, seed, weighted cross-entropy, optimizer settings, validation
interval, early stopping, and checkpoint-selection rule. The only intended
model-facing data change is:

- P2 active dates: six acquisitions ending 21 September 2026;
- P3 active dates: only 21 September 2026.

The six training rasters remain in the P3 experiment contract so the data audit,
common-validity support, split, and schedule match P2. `active_dates` controls
the tensor passed to the model and contains exactly one date.

## Initialization boundary

P3 loads the official Prithvi-EO-2.0-300M-TL pretrained encoder at the pinned
Hugging Face revision. It does not initialize from the P2 T=6 checkpoint.
The U-Net decoder and classification head start from their normal random
initialization and are trained together with the encoder using the locked
learning rates.

## Workflow

| Notebook | Responsibility |
| --- | --- |
| `11_train_p3_single_wce.ipynb` | Audit training inputs, smoke-test `[1,6,1,224,224]`, train/resume, select checkpoints |
| `21_prepare_single_date_inference_dataset.ipynb` | Audit and stage one individual-item GeoTIFF, then write a T=1 manifest |
| `31_inference_single_map_benchmark.ipynb` | Strict-load P3, benchmark on T4, and create a label-free map |

The first executable transfer target is Thái Thụy on 21 September 2026. This
allows a direct P3 T=1 versus P2 T=6 comparison with the same target date and
AOI. Protocol A can later use the single target-date GeoTIFF for Xuân Thủy.

## Model selection

The primary checkpoint rule remains maximum validation macro-F1 over all 11
fixed classes. If a gate-passing `best_accepted.pt` exists, inference uses it;
otherwise inference uses `best_any.pt` and the output remains exploratory.
Previously used points and the Thái Thụy points are not used for training,
early stopping, or checkpoint selection.

## Interpretation limits

- The training label is historical and has no confirmed 21 September date.
- No per-pixel SCL/cloud mask is supplied to the training branch.
- P3 versus P2 changes the fine-tuned temporal contract, not only inference
  latency.
- A visual difference does not identify whether the cause is pretraining,
  temporal attention, decoder learning, cloud sensitivity, or label mismatch.
- Thái Thụy point agreement is a transfer audit unless the point sampling and
  reference dates are independently documented.

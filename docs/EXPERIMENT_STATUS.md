# Experiment status

Status date: 13 October 2026.

## P2: Prithvi + UNet + WCE

Status: **completed**.

- Prithvi-EO-2.0-300M-TL pretrained encoder.
- Six dates and six bands per sample.
- Target-time token slice after joint temporal attention.
- Learned spatial pyramid, UNet decoder, and 11-class head.
- Weighted cross-entropy objective.
- 4,992 patch locations and effective batch size 16.
- Training reached step 3,120 and selected step 2,184 by validation macro-F1.
- The acceptance gate failed because only 10 of 11 classes were predicted and
  rare-class recall requirements were not met.

Selected-checkpoint validation:

| Measure | Value |
| --- | ---: |
| Macro-F1, 11 fixed classes | 0.4429241841 |
| Macro-IoU, 11 fixed classes | 0.3563422128 |
| Overall agreement | 0.8383000857 |
| WCE | 1.0932569504 |

The selected map has 1,471,132 valid pixels and is explicitly marked as not
validated against date-matched September ground truth.

Previously used reference-point agreement:

| Measure | Value |
| --- | ---: |
| Usable points | 1,036 of 1,037 |
| Overall agreement | 0.9218146718 |
| Macro-F1 over all 11 classes | 0.5747622906 |
| Macro-F1 over 7 represented classes | 0.9031978852 |

This point result is not used for checkpoint selection and must not be described
as independent September accuracy.

## Protocol A and Protocol B

Status: **workflow code and templates prepared; no transfer result claimed**.

- Protocol A keeps the Xuân Thủy region and changes the six-date sequence to
  test temporal transfer with the frozen selected P2 checkpoint.
- Protocol B changes the region while retaining a comparable season to test
  spatial transfer with the same frozen checkpoint.
- Training, inference-dataset preparation, and inference/benchmark execution
  are separated into three notebooks.
- No Protocol A or Protocol B accuracy claim is valid until the locked dataset,
  checkpoint, inference artifacts, and an independent date-matched label set
  exist. Without such labels, outputs remain qualitative transfer maps.

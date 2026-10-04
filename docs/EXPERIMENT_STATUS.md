# Experiment status

Status date: 5 October 2026.

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

## P5: frozen Prithvi warm-up to DMI + morphology

Status: **source and Colab notebook prepared; training not started**.

Planned sequence:

1. load the original Prithvi pretrained encoder;
2. freeze the encoder in evaluation mode;
3. train the task-specific neck, UNet decoder, and head with WCE;
4. check readiness every 156 steps from step 312 through step 936;
5. require two consecutive readiness passes;
6. enable exact DMI and the trainable smooth 3x3 logit-closing layer;
7. select primarily by validation macro-F1 after train-only class permutation,
   with full-rank validation DMI loss as the secondary checkpoint.

The P5 notebook does not contain experimental performance evidence. Any future
P5 number must come from its separate run directory and immutable method hash.

# Reproducibility notes

## Locked foundations

- Public helper repository commit:
  `0775770ae87b83b7d2982ed727226128685f3fba`
- Prithvi model revision:
  `63adbd39c271da4c42f447e69b1a7c91a338cdc9`
- Completed P2 method hash:
  `d4f8bef93e89450b9935fd1f982d4c3ef25d23b7386b225cb98c8e27b288021c`
- Completed WCE notebook SHA-256 before public-output cleanup:
  `fd27dcf0f3678d2ce0f21d701d104d584aa8dfd25eaf4bc1ab656d3de2c086c1`
- Prepared P5 notebook SHA-256:
  `6824eb16961b0b9324ee528684364c91a5349ac4a1e5d709af7a5b439fbac74b`

The public WCE notebook copy removes one stale earlier `NameError` output. The
later completed training, selection, map, and point-agreement outputs are left
unchanged.

## Spatial protocol

The validation core is a vertical band with a 48-pixel guard from the training
domain. Patches are 224 by 224 pixels. Validation and inference use stride 112.
The saved audit reports zero shared input pixels between the training and
validation patch domains. This spatial separation does not make pixels
independent statistical replicates.

## Checkpoint selection

P2 selection uses validation macro-F1 over 11 fixed classes. Acceptance also
requires predictions for all 11 classes and recall of at least 0.05 for source
codes 5, 6, and 7. The independent-point archive is not used for tuning or
selection.

P5 uses a separate experiment ID and run directory. It must not overwrite or
resume from the P2 run. Warm-up selection is the earliest checkpoint passing
the readiness gate twice, not the checkpoint with the highest WCE F1.

## Determinism boundary

Seeds are fixed for patch manifests, schedules, and training, but CUDA training
does not claim bitwise determinism. Report environment versions, method hashes,
checkpoint hashes, and metric tolerances with every reproduced result.

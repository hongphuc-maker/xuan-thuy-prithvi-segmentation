# Contributing

Changes should preserve the distinction between completed evidence and planned
experiments. Do not add a metric unless it is linked to a saved experiment
contract, method hash, checkpoint, and evaluation artifact.

Before opening a pull request:

1. keep private rasters, reference points, weights, and run directories out of Git;
2. update an existing immutable experiment contract only to correct metadata;
3. create a new experiment ID for changes to data, preprocessing, split, model,
   objective, optimization, or checkpoint initialization;
4. run `pytest`;
5. scan the diff for credentials and personal Drive paths.

Use a separate branch for each experiment or documentation change. Explain the
scientific comparison boundary in the pull request description.

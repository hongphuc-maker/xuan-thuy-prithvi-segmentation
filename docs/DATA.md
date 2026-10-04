# Data contract

The repository does not distribute research data. The two experiment contracts
under `configs/experiments/` define the required private files and their SHA-256
checksums.

## Current input

- Six Sentinel-2 observations: 9 January, 26 May, 18 July, 12 August,
  3 September, and 21 September 2026.
- Six bands in fixed order: B02, B03, B04, B8A, B11, B12.
- Common grid: EPSG:32648, 1,374 rows by 1,821 columns, nominal 10 m pixels.
- Target map: 21 September 2026.
- Label codes: 1 through 11, converted internally to targets 0 through 10.

The B8A, B11, and B12 source bands are originally 20 m products resampled to
the 10 m grid. Resampling does not create new 10 m observational detail.

## Radiometry and normalization

The reader applies the locked transformation:

```text
x = raw_DN - 1000
x = x * 10000 / 10000
z = (x - mean) / std
```

with:

```text
mean = [1087, 1342, 1433, 2734, 1958, 1363]
std  = [2248, 2179, 2178, 1850, 1242, 1049]
```

The -1000 offset is an explicit experiment assumption. Product metadata must be
verified before making a general scientific claim. Do not apply the correction
again to already harmonized reflectance files.

## Quality limitations

- No SCL or cloud-mask files are supplied.
- Scene-level cloud-cover filtering does not prove the area of interest is cloud-free.
- The 12 August observation contains visible cloud contamination.
- The label raster is historical and its observation date is unknown.
- Previously used reference points have no verified September date alignment.

For these reasons, the saved map remains exploratory and point comparison is
reported as agreement rather than independent September accuracy.

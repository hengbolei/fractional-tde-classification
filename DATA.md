# Data

## Source

This project uses the **MALLORN Astronomical Classification Challenge** data set:

- Challenge and download: <https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/data>
- Data paper: Dylan Magill et al., *MALLORN: many artificial LSST light curves based on observations of real nuclear transients*, *RAS Techniques and Instruments* 5 (2026), rzag019. <https://doi.org/10.1093/rasti/rzag019>

The data paper states that the MALLORN data produced for the work is available through the official Kaggle challenge. Users must review and accept the current Kaggle competition/data terms themselves. This repository intentionally does not redistribute the files.

## Download

Download the competition archive in a browser after accepting the Kaggle terms, or use the Kaggle CLI if it is already configured for your account. The Kaggle CLI is optional and is not included in `requirements.txt`.

```bash
kaggle competitions download -c mallorn-astronomical-classification-challenge
```

Extract the downloaded archive and place its contents in `mallorn-dataset/` at the repository root. The directory is ignored by Git to prevent accidental redistribution of the data.

## Expected layout

```text
mallorn-dataset/
├── train_log.csv
├── test_log.csv
├── sample_submission.csv
├── split_01/
│   ├── train_full_lightcurves.csv
│   └── test_full_lightcurves.csv
├── ...
└── split_20/
    ├── train_full_lightcurves.csv
    └── test_full_lightcurves.csv
```

The scripts currently use only `train_log.csv` and the 20 `train_full_lightcurves.csv` files because the thesis evaluates a labelled TDE-versus-non-TDE classification task.

## Validate the files

From the repository root, run either check below.

PowerShell:

```powershell
Test-Path mallorn-dataset/train_log.csv
(Get-ChildItem mallorn-dataset/split_*/train_full_lightcurves.csv).Count
```

macOS / Linux:

```bash
test -f mallorn-dataset/train_log.csv && echo "train_log.csv: OK"
find mallorn-dataset -path '*/split_*/train_full_lightcurves.csv' | wc -l
```

The split-file count should be `20`. File checksums are not included because the repository does not publish or version the upstream archive; use the files supplied by the official challenge page.

## Expected columns

`train_log.csv` must include:

- `object_id`
- redshift `Z`
- extinction `EBV`
- binary `target`

The metadata may also include the spectroscopic type and split identifier. Each light-curve CSV must include:

- `object_id`
- `Time (MJD)` (the scripts also accept `Time (MJD )`)
- `Flux`
- `Flux_err`
- `Filter`

Column names are stripped of surrounding whitespace before normalization. The code expects the standard MALLORN filter labels and does not validate a modified schema.

## Citation

```bibtex
@article{magill2026mallorn,
  title   = {MALLORN: many artificial LSST light curves based on observations of real nuclear transients},
  author  = {Magill, Dylan and Nicholl, Matt and Anilkumar, Vysakh and others},
  journal = {RAS Techniques and Instruments},
  volume  = {5},
  year    = {2026},
  pages   = {rzag019},
  doi     = {10.1093/rasti/rzag019}
}
```

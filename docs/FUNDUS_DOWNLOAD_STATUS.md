# Fundus Download Status

Source logs: `logs/download_fundus.out` and `logs/download_fundus.err` from Slurm job `228438`.

## Status from the logs

| Dataset | Status | Evidence | Next action |
| --- | --- | --- | --- |
| BRSET | Already prepared | Downloader printed that `$DATA_PATH/BRSET/brset` already has the expected prepared structure. | None. |
| mBRSET | Already prepared | Downloader printed that `$DATA_PATH/mBRSET/mbrset` already has the expected prepared structure. | None. |
| RFMiD | Downloaded by Kaggle | Kaggle command completed; final failure list does not include RFMiD. | Inspect folder structure and labels. |
| RFMiD 2.0 | Downloaded and extracted | Zenodo zip downloaded and extracted. | Inspect folder structure and labels. |
| Messidor-2 | Downloaded by Kaggle | Kaggle command completed; final failure list does not include Messidor-2. | Inspect folder structure and labels. |
| G1020 | Downloaded by Kaggle | Kaggle command completed; final failure list does not include G1020. | Inspect because the Kaggle mirror contains multiple glaucoma datasets. |
| JSIEC1000 | Downloaded by Kaggle | Kaggle command completed; final failure list does not include JSIEC1000. | Inspect folder structure and labels. |
| PAPILA | Failed | `list indices must be integers or slices, not str` from Figshare DOI resolution. | Fixed in `scripts/download_fundus_datasets.py`; rerun only `papila`. |
| IDRiD | Not downloaded | Script wrote `IDRiD/MANUAL_DOWNLOAD.md`. | Manual download from IEEE DataPort, or add a programmatic official source if one is approved. |
| REFUGE | Optional/manual | Script wrote `REFUGE/MANUAL_DOWNLOAD.md`. A REFUGE copy is bundled inside the G1020 Kaggle mirror, but the current benchmark uses only the G1020 subset. | Add manually only if REFUGE should be evaluated as its own dataset. |
| PALM | Not downloaded | Script wrote `PALM/MANUAL_DOWNLOAD.md`. | Manual download after Grand Challenge registration/terms acceptance. |

## Inspecting the real Lustre structure

Because this repo may be mounted over SSHFS, use the CPU inspection job instead of walking Lustre locally:

```bash
sbatch jobs/download/inspect_fundus_datasets.sh
```

The report is written to:

```text
logs/inspect_fundus.out
logs/inspect_fundus.err
```

The inspection job reports, for each benchmark dataset:

- top-level directory structure;
- total image files;
- total CSV files;
- zip files;
- manual-download notes;
- sample CSV paths;
- largest image-containing directories.

## Preprocessing

`scripts/download_fundus_datasets.py` separates raw download from preprocessing. Reruns now follow this order:

1. skip the raw download if dataset files already exist;
2. check for `prepared/manifest.csv`;
3. build only the missing preprocessing output.

To preprocess already downloaded data without attempting downloads:

```bash
sbatch jobs/download/download_fundus_datasets.sh --preprocess-only
```

The standardized output is:

```text
$DATA_PATH/<dataset>/prepared/manifest.csv
```

BRSET and mBRSET keep their existing loader-compatible structures and also get a manifest.

## Rerunning only missing automatic data

After the PAPILA Figshare fix, rerun the downloader normally or rerun just PAPILA. The downloader now skips any dataset directory that already contains real data files, so rerunning the full job is safe.

```bash
sbatch jobs/download/download_fundus_datasets.sh --datasets papila
```

or:

```bash
sbatch jobs/download/download_fundus_datasets.sh
```

Manual datasets should be handled separately:

```text
$DATA_PATH/IDRiD
$DATA_PATH/PALM
```

REFUGE is intentionally excluded from the default included set for now. The G1020 Kaggle mirror contains a `G1020/REFUGE` folder, but that folder is not counted as standalone REFUGE unless we add a dedicated REFUGE preprocessing path.

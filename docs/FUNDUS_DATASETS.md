# Fundus Dataset Audit

This benchmark is scoped to color fundus photography. The primary inclusion rule is:

- include open or practically obtainable fundus datasets that are not documented as pretraining data for the retinal foundation models we evaluate;
- keep datasets without demographics for performance-only evaluation;
- run demographic or bias analyses only on datasets where demographic metadata is available.

## Included datasets

| Dataset | Region | Main use | Demographics | Access/download | Notes |
| --- | --- | --- | --- | --- | --- |
| BRSET | Brazil | Multilabel disease and demographic prediction | Age, sex, nationality, diabetes history | PhysioNet credentialed | Current benchmark dataset. |
| mBRSET | Brazil | Clinical and demographic prediction from portable fundus photos | Available demographic/clinical variables | PhysioNet credentialed | Current benchmark dataset. |
| PAPILA | Spain | Glaucoma classification, optic disc/cup segmentation | Age, sex | Figshare DOI | Strongest non-Brazilian open fundus dataset with demographics. |
| RFMiD | India | Multi-disease classification | Not public | Kaggle mirror or RIADD/Grand Challenge | Broad pathology coverage. |
| RFMiD 2.0 | India | Multi-disease classification | Not public | Zenodo | Auxiliary RFMiD release. |
| IDRiD | India | DR/DME grading and lesion segmentation | Not public | IEEE DataPort open-access page | Use official source for audit; script leaves this manual. |
| Messidor-2 | France | DR/DME grading | Not public | Consortium/form or Kaggle mirror | Good DR benchmark; keep license/source caveat visible. |
| REFUGE | China / challenge | Glaucoma classification, disc/cup segmentation, fovea localization | Not public | G1020 Kaggle mirror / Grand Challenge | Prepared as a standalone glaucoma dataset from the bundled REFUGE copy. |
| G1020 | Germany / clinical | Glaucoma classification and disc/cup segmentation | Not public | Kaggle mirror | Use the G1020 subset from combined mirrors. |
| JSIEC1000 | China | 39-class fundus disease classification | Not public | Kaggle | Good disease breadth slice. |
| PALM | China | Pathologic myopia and lesion segmentation | Not public | Grand Challenge registration | Include if myopia is in scope. |


## Excluded datasets

| Dataset | Reason excluded |
| --- | --- |
| EyePACS / Kaggle DR | Explicitly documented as RETFound color-fundus pretraining data. |
| AIROGS | Documented in public retinal foundation-model pretraining mixtures. |
| DDR | Documented in public retinal foundation-model pretraining mixtures. |
| ODIR-5K / ODIR-2019 | Has age and sex, but documented in public retinal foundation-model pretraining mixtures. |
| Harvard-FairVLMed / Harvard-FairSeg | Useful fairness data, but SLO rather than standard color fundus photography. |
| UK Biobank retinal imaging | Strong demographics, but controlled access rather than open benchmark download. |
| AREDS / AREDS2 | Strong AMD cohort, but controlled dbGaP-style access rather than open benchmark download. |
| NHANES raw retinal images | Demographic-rich survey, but raw images are not a simple open download. |
| Singapore Epidemiology of Eye Diseases | Strong multiethnic cohort, but requires collaboration/access approval. |
| MEH-MIDAS | Not open and documented in RETFound-style pretraining. |
| SDPP | Not open and documented in retinal foundation-model pretraining. |

## Downloading

The manifest is in `config/fundus_datasets.yaml`. The downloader is:

```bash
python3 scripts/download_fundus_datasets.py --config config/fundus_datasets.yaml
```

By default it downloads datasets marked `included` directly into `$DATA_PATH` using `config/paths.env`. Optional secondary datasets are skipped unless `--include-optional` is passed.

BRSET and mBRSET are normalized to match the existing `RetinaDataset` loader:

```text
$DATA_PATH/BRSET/brset/labels_brset.csv
$DATA_PATH/BRSET/brset/labels.csv
$DATA_PATH/BRSET/brset/images_224/

$DATA_PATH/mBRSET/mbrset/labels_mbrset.csv
$DATA_PATH/mBRSET/mbrset/labels.csv
$DATA_PATH/mBRSET/mbrset/images_224/
```

If those expected files/directories already exist, the downloader skips the dataset.

For PhysioNet downloads, put credentials in `.env`:

```bash
PHYSIONET_USERNAME=your_physionet_username
PHYSION_PASSWORD=...
```

Kaggle mirrors require the Kaggle CLI credentials file, usually `~/.kaggle/kaggle.json`. Grand Challenge and some official sources require web-form or account acceptance; for those, the script writes a `MANUAL_DOWNLOAD.md` note in the expected destination directory.

Some sources require credentials or web-form acceptance. The script creates a `MANUAL_DOWNLOAD.md` file inside each such dataset directory with the exact source URL and expected location.

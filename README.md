# FOCUS

![FOCUS logo](logo/FOCUS-logo.png)

FOCUS is an anonymous retinal foundation-model benchmark for evaluating computer vision models, vision-language models, and multimodal large language models on color fundus tasks. The benchmark covers zero-shot evaluation, linear probing, calibration, subgroup analysis, and LoRA adaptation with supervised fine-tuning (SFT) and GRPO.

![FOCUS benchmark overview](logo/benchmark_overview.png)

An interactive arena is available at:

https://huggingface.co/spaces/focus-retina-benchmark/focus-retinal-benchmark-arena

This repository is prepared for anonymous review. Do not add author names, personal emails, institution-specific notes, private paths, credentials, raw datasets, model weights, or result dumps to the public repository.

## Repository Layout

```text
retina_bench/
  core/          dataset loading, preprocessing helpers, shared metrics
  cv/            CV foundation-model download and evaluation
  vlms/          CLIP/SigLIP/ophthalmic VLM download and evaluation
  mllms/         MLLM prompts, download, and first-token evaluation
  adaptation/    SFT and GRPO LoRA training scripts

config/
  paths.env.template      path template copied to config/paths.env
  fundus_datasets.yaml    dataset manifest and preprocessing rules
  cv_config.yaml          CV model, dataset, task, and method grid
  vlms_config.yaml        VLM model, dataset, task, and method grid
  mllm_config.yaml        MLLM evaluation and training grid
  analysis_config.yaml    model groups, metrics, and analysis settings

jobs/
  setup/         reproducibility export jobs
  download/      online download/cache jobs
  eval/          offline benchmark and analysis jobs
  train/         offline SFT and GRPO jobs

Split_Data/
  labels_splits_brset.csv
  labels_splits_mbrset.csv

ext_repos/
  patched external model code used by selected foundation models
```

## Pipeline

1. Create the environment.
2. Configure local paths and credentials.
3. Run online download jobs for datasets and models.
4. Run offline CV, VLM, and MLLM benchmark jobs.
5. Optionally train SFT/GRPO adapters.
6. Evaluate adapters offline.
7. Run analysis jobs to aggregate performance, calibration, fairness, and robustness metrics.

The download jobs run with online access. The benchmark, training, and analysis jobs are configured for offline execution through the Hugging Face cache and local dataset/model directories.

## 1. Environment

The benchmark can be installed from a fresh conda environment, from the exported environment file, or with the provided Slurm helper script.

Option A: create a fresh Python 3.12 environment and install the benchmark dependencies:

```bash
conda create -n llms python=3.12 -y
conda activate llms

python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -r requirements.txt
```

Option B: recreate the exported conda environment directly:

```bash
conda env create -f reproducibility/environment.lock.yml
conda activate llms
```

Option C: use the provided Slurm installation script:

```bash
sbatch install_packages.sh
```

The Slurm scripts assume a module-based HPC environment and may need local edits to the `#SBATCH` headers or module names. `install_packages.sh` accepts environment overrides such as `CONDA_ENV_NAME`, `PYTHON_VERSION`, `MINIFORGE_MODULE`, `CUDA_MODULE`, `CUDNN_MODULE`, `NCCL_MODULE`, and `SKIP_MODULES=1`.

`requirements.txt` intentionally excludes `torch`, `torchvision`, `torchaudio`, and the `nvidia-*-cu12` runtime wheels because PyTorch should be installed first with the CUDA 12.4 wheel command above. Some download jobs install small helper packages such as `kaggle`, `gdown`, `pillow`, or editable external repos when needed.

## 2. Paths And Credentials

Create the local path file:

```bash
cp config/paths.env.template config/paths.env
```

Edit `config/paths.env`:

```bash
HF_HOME="$WORK/.cache/huggingface"
DATA_PATH="/path/to/fundus/datasets"
OUTPUT_DIR="results/evals"
CHECKPOINTS_DIR="results/checkpoints"
EXT_REPOS_DIR="ext_repos"
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`config/paths.env` is intentionally ignored by git because it contains machine-specific paths.

Create the local credentials file:

```bash
cp .env.example .env
```

Fill only the credentials you need:

```bash
HF_TOKEN=...
PHYSIONET_USERNAME=...
PHYSIONET_PASSWORD=...
```

Credential use:

- `HF_TOKEN`: required for gated Hugging Face models.
- `PHYSIONET_USERNAME` and `PHYSIONET_PASSWORD`: required for BRSET and mBRSET.
- Kaggle datasets require the Kaggle CLI credential file, usually `~/.kaggle/kaggle.json`.
- Manual/web-form datasets are documented in `docs/FUNDUS_DATASETS.md` and `config/fundus_datasets.yaml`.

Never commit `.env`, `config/paths.env`, Kaggle credentials, service-account JSON files, raw datasets, or model weights.

## 3. Dataset Manifest

Datasets are defined in `config/fundus_datasets.yaml`. Each dataset entry controls:

- `status`: included datasets are downloaded by default.
- `download.method`: `physionet`, `kaggle`, `zenodo`, `figshare`, or `manual`.
- `download.storage_dir`: destination under `$DATA_PATH`.
- `download.prepared_paths`: files that must exist for preprocessing to be considered complete.
- `download.prepare`: preprocessing routine used by `scripts/download_fundus_datasets.py`.
- `evaluation`: optional fairness and robustness metadata.

The benchmark currently uses:

```text
brset, mbrset, papila, rfmid, rfmid_2, idrid, messidor_2, g1020, jsiec1000
```

BRSET and mBRSET require fixed benchmark splits. The repository includes:

```text
Split_Data/labels_splits_brset.csv
Split_Data/labels_splits_mbrset.csv
```

During preprocessing, the downloader copies them if missing:

```text
$DATA_PATH/BRSET/brset/labels_splits.csv
$DATA_PATH/mBRSET/mbrset/labels_splits.csv
```

These are the filenames expected by `retina_bench/core/data.py`.

## 4. Download Datasets

Run downloads on a node with internet access:

```bash
sbatch jobs/download/download_fundus_datasets.sh
```

Common variants:

```bash
sbatch jobs/download/download_fundus_datasets.sh --datasets brset mbrset
sbatch jobs/download/download_fundus_datasets.sh --download-only
sbatch jobs/download/download_fundus_datasets.sh --preprocess-only
sbatch jobs/download/download_fundus_datasets.sh --force-preprocess
sbatch jobs/download/download_fundus_datasets.sh --dry-run
```

The script downloads into `$DATA_PATH`, prepares normalized labels/manifests, creates `images_224` where needed, and writes prepared manifests under each dataset directory. See `docs/FUNDUS_DATASETS.md` for dataset access notes.

## 5. Download Models For Offline Use

The compute jobs run with:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Therefore all Hugging Face models, Google Drive weights, TorchVision weights, CLIP weights, and patched external-repo weights must be downloaded before evaluation/training.

Run the online cache jobs:

```bash
sbatch jobs/download/download_mllms.sh
sbatch jobs/download/download_vlms.sh
sbatch jobs/download/download_cv.sh
```

The MLLM downloader reads `config/mllm_config.yaml`. The VLM and CV downloaders cache Hugging Face models and selected external weights needed by code under `ext_repos/`.

Important: `ext_repos/` contains patched model code used by the benchmark and should be versioned as source. The large weights inside `ext_repos/`, such as `.pt` and `.pth` files, are ignored and should be downloaded locally.

## 6. YAML Configuration Files

The Slurm jobs read YAML files dynamically. To change the benchmark grid, edit YAML rather than editing loops in shell scripts.

`config/mllm_config.yaml` controls MLLM evaluation and adapter training:

- `models`: Hugging Face model ids.
- `datasets`: datasets considered during baseline evaluation.
- `tasks`: global task list.
- `dataset_tasks`: allowed task subset per dataset.
- `train_datasets`: datasets used for SFT/GRPO.
- `train_tasks`: tasks used for SFT/GRPO.
- `strategies`: prompt strategies for baseline/adapted evaluation.
- `sft_strategies`: prompt strategies used for SFT training.
- `grpo_strategies`: prompt strategies used for GRPO training.
- `quantization`: quantization mode passed to MLLM scripts.
- `use_unsloth`: switches training scripts to the Unsloth variants when true.
- `eval_datasets_ft` and `eval_tasks_ft`: optional out-of-domain adapter evaluation grid.

`config/vlms_config.yaml` controls CLIP/SigLIP/ophthalmic VLM evaluation:

- `models`: Hugging Face and external VLM names.
- `datasets`, `tasks`, `dataset_tasks`: evaluation grid.
- `eval_methods`: for example `zero_shot` and `linear_probing`.

`config/cv_config.yaml` controls image-only foundation-model evaluation:

- `models`: general and ophthalmic CV encoders.
- `datasets`, `tasks`, `dataset_tasks`: evaluation grid.
- `eval_methods`: currently linear probing by default.
- `pooling`: `cls` or `gap`.

`config/analysis_config.yaml` controls aggregation:

- `model_groups`: groups used in tables/figures.
- `metrics`: performance, calibration, fairness, and robustness metrics.
- `fairness_attributes`: dataset-specific demographic columns.
- `datasets`, `tasks`, `dataset_tasks`: analysis coverage.

`config/fundus_datasets.yaml` controls dataset download and preparation.

## 7. Run Baseline Benchmarks

After datasets and models are cached, submit the offline evaluation jobs:

```bash
sbatch jobs/eval/run_cv_benchmark.sh
sbatch jobs/eval/run_vlms_benchmark.sh
sbatch jobs/eval/run_mllms_benchmark.sh
```

Outputs are written under `$OUTPUT_DIR`, normally `results/evals`. The `.gitignore` excludes results because they are generated artifacts.

## 8. Train Adapters

SFT and GRPO are driven by `config/mllm_config.yaml` using `models`, `train_datasets`, `train_tasks`, `sft_strategies`, `grpo_strategies`, `quantization`, and `use_unsloth`.

Run SFT:

```bash
sbatch jobs/train/run_sft.sh
```

Run GRPO:

```bash
sbatch jobs/train/run_grpo.sh
```

Adapters are written under `$CHECKPOINTS_DIR`, normally:

```text
results/checkpoints/sft_<model>_<dataset>_<task>_<strategy>
results/checkpoints/grpo_<model>_<dataset>_<task>_<strategy>
```

These checkpoints are generated artifacts and should not be committed.

## 9. Evaluate Adapters

Evaluate trained SFT adapters:

```bash
sbatch jobs/eval/evaluate_sft_adapters.sh
```

Evaluate trained GRPO adapters:

```bash
sbatch jobs/eval/evaluate_grpo_adapters.sh
```

The adapter evaluation jobs skip missing checkpoint directories, so partial training grids are supported.

## 10. Analyze Results

Aggregate baseline results:

```bash
sbatch jobs/eval/run_analysis.sh
```

Aggregate fine-tuned adapter results:

```bash
sbatch jobs/eval/run_analysis_ft.sh
```

Analysis outputs are written to `results/analysis` and `results/analysis_ft`. Manuscript table/figure outputs are generated artifacts and are ignored by default.

## Direct Script Entry Points

Most users should submit Slurm jobs. For debugging or local use, the underlying Python entry points are:

```bash
python scripts/download_fundus_datasets.py --help
python retina_bench/cv/download.py --help
python retina_bench/vlms/download.py --help
python retina_bench/mllms/download.py --help

python retina_bench/cv/evaluate.py --help
python retina_bench/vlms/evaluate.py --help
python retina_bench/mllms/evaluate.py --help

python retina_bench/adaptation/train_sft.py --help
python retina_bench/adaptation/train_grpo.py --help
```

Set `PYTHONPATH` from the repository root when running scripts directly:

```bash
export PYTHONPATH="${PYTHONPATH:-}:."
```

## Programmatic Use

The benchmark modules can also be used directly when you want to build custom experiments, reuse the curated fundus loaders, or extract model embeddings outside the Slurm evaluation grid.

### Load A Dataset

`RetinaDataset` normalizes the supported fundus datasets into a common row interface. `base_dir` should be the same path used as `DATA_PATH` in `config/paths.env`.

```python
from retina_bench.core.data import RetinaDataset

dataset = RetinaDataset(
    base_dir="/path/to/fundus/datasets",
    dataset_name="brset",
    split="test",
)

print(len(dataset))
row = dataset.get_row(0)
image = dataset.get_image(0)

print(row[["image_id", "split", "DR_2_Class", "Task_Referable"]])
print(image.size)
```

For BRSET and mBRSET, the loader expects the split files prepared by `scripts/download_fundus_datasets.py`:

```text
$DATA_PATH/BRSET/brset/labels_splits.csv
$DATA_PATH/mBRSET/mbrset/labels_splits.csv
```

### Extract Image Embeddings From CV Models

Image-only foundation models live under `retina_bench.cv`. They expose `get_image_embeddings(images)`.

```python
import torch

from retina_bench.core.data import RetinaDataset
from retina_bench.cv.models import get_cv_model

data = RetinaDataset("/path/to/fundus/datasets", "papila", split="test")
images = [data.get_image(i) for i in range(4)]

model = get_cv_model(
    "facebook/dinov2-large",
    device="cuda",
    pooling="cls",  # or "gap"
)

with torch.no_grad():
    image_embeddings = model.get_image_embeddings(images)

print(image_embeddings.shape)
```

The same interface is used for supported ophthalmic CV encoders such as RETFound and VisionFM, provided their weights have already been downloaded:

```python
model = get_cv_model("YukunZhou/RETFound_mae_natureCFP", device="cuda")
embeddings = model.get_image_embeddings(images)
```

### Extract Image And Text Embeddings From VLMs

Vision-language models live under `retina_bench.vlms`. They expose both image and text embedding methods, which makes them useful for retrieval, custom zero-shot classifiers, and embedding export.

```python
import torch

from retina_bench.core.data import RetinaDataset
from retina_bench.vlms.models import get_vlm_model

data = RetinaDataset("/path/to/fundus/datasets", "brset", split="test")
images = [data.get_image(i) for i in range(8)]

texts = [
    "a fundus photograph with diabetic retinopathy",
    "a fundus photograph without diabetic retinopathy",
]

model = get_vlm_model("google/medsiglip-448", device="cuda")

with torch.no_grad():
    image_embeddings = model.get_image_embeddings(images)
    text_embeddings = model.get_text_embeddings(texts)
    logits = image_embeddings @ text_embeddings.T
    probabilities = logits.softmax(dim=-1)

print(probabilities)
```

External ophthalmic VLMs use the same methods:

```python
model = get_vlm_model("EyeCLIP", device="cuda")
image_embeddings = model.get_image_embeddings(images)
text_embeddings = model.get_text_embeddings(texts)
```

### Run MLLM Inference

MLLM wrappers live in `retina_bench.mllms.models`. The helper below mirrors the class selection used by the benchmark evaluator.

```python
from retina_bench.core.data import RetinaDataset
from retina_bench.mllms.evaluate import get_model_class
from retina_bench.mllms import prompts

dataset_name = "brset"
task = "referable_dr"
model_id = "google/medgemma-4b-it"

data = RetinaDataset("/path/to/fundus/datasets", dataset_name, split="test")
row = data.get_row(0)
image = data.get_image(0)

prompt_func = prompts.get_prompt_func(dataset_name, task, "base")
prompt = prompt_func(row)

ModelClass = get_model_class(model_id)
model = ModelClass(
    model_id=model_id,
    quantization="16b",
    use_flash_attention=True,
)

output = model.generate(
    prompt=prompt,
    image=image,
    max_new_tokens=32,
    do_sample=False,
)

print(output["text"])
```

For calibrated binary classification, you can request first-token scores for task labels when the underlying wrapper supports returning generation scores:

```python
output = model.generate(
    prompt=prompt,
    image=image,
    max_new_tokens=4,
    return_logits=True,
    tokens=["yes", "no", "Yes", "No"],
    do_sample=False,
)

print(output.get("token_probs", {}))
```

All model examples assume the required checkpoints are already present in the Hugging Face cache or under `ext_repos/`, because the benchmark is designed to run offline after the download stage.

## Notes

- Dataset access and model terms must be accepted upstream before running the download jobs. Useful access links:

  Dataset sources:

  - BRSET: https://physionet.org/content/brazilian-ophthalmological/1.0.0/
  - mBRSET: https://physionet.org/content/mbrset/1.0/
  - PAPILA: https://figshare.com/articles/dataset/PAPILA/14798004
  - RFMiD: https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification
  - RFMiD 2.0: https://zenodo.org/records/7505822
  - IDRiD mirror used by the downloader: https://zenodo.org/records/17219542
  - Messidor-2 Kaggle mirror used by the downloader: https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess
  - G1020 Kaggle mirror used by the downloader: https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets
  - JSIEC1000 Kaggle mirror used by the downloader: https://www.kaggle.com/datasets/linchundan/fundusimage1000

  Hugging Face model pages:

  - Gemma 3 27B: https://huggingface.co/google/gemma-3-27b-it
  - MedGemma 4B: https://huggingface.co/google/medgemma-4b-it
  - MedGemma 1.5 4B: https://huggingface.co/google/medgemma-1.5-4b-it
  - MedGemma 27B: https://huggingface.co/google/medgemma-27b-it
  - Qwen3-VL 8B: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
  - LLaVA-Llama3 8B: https://huggingface.co/llava-hf/llama3-llava-next-8b-hf
  - MedSigLIP: https://huggingface.co/google/medsiglip-448
  - SigLIP2: https://huggingface.co/google/siglip2-base-patch16-224
  - CLIP ViT-B/32: https://huggingface.co/openai/clip-vit-base-patch32
  - FLAIR: https://huggingface.co/jusiro2/FLAIR
  - ViT-L/16: https://huggingface.co/google/vit-large-patch16-224
  - DINOv2 Large: https://huggingface.co/facebook/dinov2-large
  - DINOv3 ViT-L/16: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
  - RETFound MAE Nature CFP: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
  - RETFound MAE Nature OCT: https://huggingface.co/YukunZhou/RETFound_mae_natureOCT
  - RETFound MAE MEH: https://huggingface.co/YukunZhou/RETFound_mae_meh
  - RETFound MAE Shanghai: https://huggingface.co/YukunZhou/RETFound_mae_shanghai
  - RETFound DINOv2 MEH: https://huggingface.co/YukunZhou/RETFound_dinov2_meh
  - RETFound DINOv2 Shanghai: https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai

- Dataset access terms remain the responsibility of the user. The downloader automates retrieval only where credentials or public mirrors allow it.
- Some model checkpoints are gated and require accepting upstream terms before `download_*` jobs can cache them.
- The benchmark expects prepared data under `$DATA_PATH` and cached models under `$HF_HOME`; evaluation and training jobs intentionally run offline.

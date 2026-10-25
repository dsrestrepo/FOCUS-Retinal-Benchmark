# FOCUS

Official implementation of **FOCUS: Benchmarking Retinal Model Generalization from Foundation Vision Encoders to Multimodal LLMs**.

<img src="logo/FOCUS-logo.png" alt="FOCUS logo" width="50%">

## Accepted at NeurIPS 2026, Evaluations & Datasets Track. 🎉

David Restrepo, Chenwei Wu, Luis Filipe Nakayama, Miguel L. Martins, Stergios Christodoulidis, Maria Vakalopoulou, and Enzo Ferrante.

[Paper (arXiv)](https://arxiv.org/abs/2609.33158) · [NeurIPS poster page](https://neurips.cc/virtual/2026/poster/139808) · [Interactive Arena](https://huggingface.co/spaces/dsrestrepo/FOCUS-Retinal-Benchmark)

## Citation

If you use FOCUS, please cite the paper:

<details>
<summary><strong>BibTeX citation</strong></summary>

```bibtex
@misc{restrepo2026focus,
  title         = {{FOCUS}: Benchmarking Retinal Model Generalization from Foundation Vision Encoders to Multimodal {LLMs}},
  author        = {Restrepo, David and Wu, Chenwei and Nakayama, Luis Filipe and Martins, Miguel L. and Christodoulidis, Stergios and Vakalopoulou, Maria and Ferrante, Enzo},
  year          = {2026},
  eprint        = {2609.33158},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2609.33158},
  url           = {https://arxiv.org/abs/2609.33158},
  note          = {Accepted at NeurIPS 2026, Evaluations \& Datasets Track}
}
```

</details>

FOCUS is a retinal foundation-model benchmark for evaluating computer vision models, vision-language models, and multimodal large language models on color fundus tasks. The benchmark covers zero-shot evaluation, linear probing, calibration, subgroup analysis, and LoRA adaptation with supervised fine-tuning (SFT) and GRPO. The paper reports baseline evaluations and SFT adaptation; GRPO is an optional extension.

![FOCUS benchmark overview](logo/benchmark_overview.png)

## Clone the Repository

```bash
git clone https://github.com/dsrestrepo/FOCUS-Retinal-Benchmark.git
cd FOCUS-Retinal-Benchmark
mkdir -p logs
```

Run the commands below from this directory. Patched external model source code is included under `ext_repos/`; dataset files and model weights are downloaded separately.

<details>
<summary><strong>Repository layout</strong></summary>

```text
retina_bench/
  core/          dataset loading, shared metrics, common utilities
  cv/            CV foundation-model download and evaluation
  vlms/          CLIP/SigLIP/ophthalmic VLM download and evaluation
  mllms/         MLLM prompts, download, and first-token evaluation
  adaptation/    SFT and GRPO LoRA training scripts

config/
  paths.env.template      path template copied to config/paths.env
  fundus_datasets.yaml    dataset registry used by the benchmark
  cv_config.yaml          CV model, dataset, task, and method grid
  vlms_config.yaml        VLM model, dataset, task, and method grid
  mllm_config.yaml        MLLM evaluation and training grid
  analysis_config.yaml    model groups, metrics, and analysis settings

jobs/
  setup/         reproducibility export jobs
  download/      online dataset/model cache jobs
  eval/          offline benchmark and analysis jobs
  train/         offline SFT and GRPO jobs

Split_Data/
  benchmark split definitions used by the dataset loader

ext_repos/
  patched external model code used by selected foundation models
```

</details>

<details>
<summary><strong>Workflow overview</strong></summary>

1. Create the environment.
2. Configure local paths and credentials.
3. Download and prepare datasets from the original sources.
4. Cache required models.
5. Run CV, VLM, and MLLM benchmark jobs.
6. Optionally train SFT/GRPO adapters.
7. Evaluate adapters.
8. Run analysis jobs to aggregate performance, calibration, fairness, and robustness metrics.

The benchmark, training, and analysis jobs are configured for offline execution through the Hugging Face cache and local dataset/model directories.

</details>

`sbatch` returns a job ID before the job finishes. Complete data/model downloads before submitting evaluation jobs, complete training before adapter evaluation, and complete evaluation before analysis. For the paper workflow, use the baseline jobs and SFT jobs below; GRPO jobs are optional.

## Environment

Run the setup commands from the repository root. The release environment targets
Linux x86_64 with NVIDIA GPUs and Python 3.12. It uses `torch==2.10.0` and
`torchvision==0.25.0`, the versions recorded in the experiment pip export.
The PyPI Linux wheels include the matching CUDA 12.8 runtime dependencies.

FlashAttention is installed after PyTorch and its build prerequisites. Building it
requires a CUDA toolkit; use CUDA 12.8 for this environment.

Option A: create a fresh environment and install the resolved benchmark dependencies:

```bash
conda create -n llms python=3.12.12 pip=25.2 -y
conda activate llms
python -m pip install -r reproducibility/requirements.lock.txt
python -m pip install --no-deps -e ./ext_repos/FLAIR
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
```

<details>
<summary><strong>Alternative installation methods and environment details</strong></summary>

Option B: create the same environment from the curated Conda file:

```bash
conda env create -f reproducibility/environment.lock.yml
conda activate llms
python -m pip install --no-deps -e ./ext_repos/FLAIR
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
```

Option C: use the provided Slurm installation script:

```bash
mkdir -p logs
sbatch install_packages.sh
```

The Slurm scripts assume a module-based HPC environment. Adjust the `#SBATCH`
headers and module names for your cluster. `install_packages.sh` accepts
`CONDA_ENV_NAME`, `PYTHON_VERSION`, `MINIFORGE_MODULE`, `GCC_MODULE`, `CUDA_MODULE`,
`MAX_JOBS`, and `SKIP_MODULES=1`. Its default CUDA module is `cuda/12.8.1`;
PyTorch supplies its own matching cuDNN and NCCL runtime packages.

`requirements.txt` lists the benchmark's direct dependencies.
`reproducibility/requirements.lock.txt` pins their resolved dependencies for
Linux/Python 3.12, and the Conda file contains the same package set. Accelerate
and OpenAI CLIP use the exact Git commits from the recorded pip export, because
their exported version strings are not installable PyPI releases.

The default environment covers VM/VLM/MLLM evaluation, data preparation, analysis,
and standard Transformers/TRL SFT and GRPO. Unsloth was not used in the paper
experiments and is excluded; keep `use_unsloth: false` for this environment.
The optional Unsloth entrypoints and external API wrappers require separate
additional dependencies. TorchAudio, xFormers, and unrelated experimental
packages are also excluded from the release install lists.

The original pip freeze, Conda package inventory, and environment report are kept
locally and excluded from Git. They describe a broader environment, including
unused packages and recorded dependency conflicts. The public
`reproducibility/python-runtime.json` records the original Python/platform
information. New exports from `jobs/setup/export_reproducibility.sh` default to
the ignored `reproducibility/snapshots/<UTC timestamp>/` directory so they do not
overwrite the curated release files.

</details>

## Paths And Credentials

Create the local path file:

```bash
cp config/paths.env.template config/paths.env
```

Edit `config/paths.env`:

```bash
HF_HOME="$WORK/.cache/huggingface"
DATA_PATH="/path/to/prepared-fundus-datasets"
OUTPUT_DIR="results/evals"
CHECKPOINTS_DIR="results/checkpoints"
EXT_REPOS_DIR="ext_repos"
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The `$WORK` example assumes your HPC environment defines that variable. Otherwise, set `HF_HOME` to an absolute cache directory.

`config/paths.env` is intentionally ignored by git because it contains machine-specific paths.

After loading that file, jobs derive the Hugging Face Hub cache as `$HF_HOME/hub`.
Adapter evaluation and analysis directories default to siblings of `$OUTPUT_DIR`.
For example, `OUTPUT_DIR="/scratch/focus/evals"` uses `/scratch/focus/evals_sft`,
`/scratch/focus/evals_grpo`, `/scratch/focus/analysis`, and
`/scratch/focus/analysis_ft`. Optional directory overrides are listed in
`config/paths.env.template`. The default `OUTPUT_DIR="results/evals"` preserves
the existing `results/` layout. Path resolution does not move existing files.

If you use gated Hugging Face models or download datasets from credentialed sources, create the local credentials file:

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
- `PHYSIONET_USERNAME` and `PHYSIONET_PASSWORD`: required if downloading BRSET and mBRSET from PhysioNet.
- Kaggle datasets require the Kaggle CLI credential file, usually `~/.kaggle/kaggle.json`.
- Manual/web-form datasets are documented in `docs/FUNDUS_DATASETS.md` and `config/fundus_datasets.yaml`.

Never commit `.env`, `config/paths.env`, Kaggle credentials, service-account JSON files, raw datasets, or model weights.

## Download and Prepare Benchmark Data

Set `DATA_PATH` in `config/paths.env` to the directory where the prepared dataset folders should live:

```bash
DATA_PATH="/path/to/prepared-fundus-datasets"
```

Datasets are defined in `config/fundus_datasets.yaml`. The benchmark currently uses:

```text
brset, mbrset, papila, rfmid, rfmid_2, idrid, messidor_2, refuge, g1020, jsiec1000
```

Run downloads on a node with internet access:

```bash
sbatch jobs/download/download_fundus_datasets.sh
```

<details>
<summary><strong>Download options and benchmark split files</strong></summary>

Common variants:

```bash
sbatch jobs/download/download_fundus_datasets.sh --datasets brset mbrset
sbatch jobs/download/download_fundus_datasets.sh --download-only
sbatch jobs/download/download_fundus_datasets.sh --preprocess-only
sbatch jobs/download/download_fundus_datasets.sh --force-preprocess
sbatch jobs/download/download_fundus_datasets.sh --dry-run
```

The dataset downloader writes benchmark-ready dataset folders under `$DATA_PATH`.

Benchmark split definitions are included in the repository under `Split_Data/` and are used by `retina_bench.core.data.RetinaDataset` when constructing train, validation, test, or all splits. For BRSET and mBRSET, the downloader installs the corresponding split file into the prepared dataset directory; for the manifest-based datasets, the loader can read the shared split definitions directly. Users do not need to create split files to run the experiments.

</details>

## Download Models For Offline Use

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

<details>
<summary><strong>Advanced YAML configuration</strong></summary>

The Slurm jobs read YAML files dynamically. To change the benchmark grid, edit YAML rather than editing loops in shell scripts.

`config/mllm_config.yaml` controls MLLM evaluation and adapter training:

- `models`: Hugging Face model ids.
- `datasets`: datasets considered during baseline evaluation. Supported benchmark
  names are `brset`, `mbrset`, `papila`, `rfmid`, `rfmid_2`, `idrid`,
  `messidor_2`, `refuge`, `g1020`, and `jsiec1000`.
- `tasks`: global task list. The harmonized binary tasks are `binary_dr`,
  `referable_dr`, and `glaucoma`.
- `dataset_tasks`: allowed task subset for each dataset.
- `train_datasets`: datasets used for SFT/GRPO. Adapter training supports
  `brset`, `mbrset`, or `both`; `both` pools the two training datasets into one
  adapter.
- `train_tasks`: tasks used for SFT/GRPO. Values are `binary_dr`,
  `referable_dr`, `glaucoma`, or `all`; `all` trains one adapter across all
  tasks available in the training dataset.
- `strategies`: prompt strategies for baseline/adapted evaluation: `base`,
  `role`, or `cot`.
- `sft_strategies`: prompt strategies used for SFT training: `base`, `role`,
  `cot`, or `all`. With `all`, base, role, and CoT formulations are included in
  one adapter.
- `grpo_strategies`: prompt strategies used for GRPO training: `base_grpo`,
  `role_grpo`, `cot`, or `all`. With `all`, all three formulations are used.
- `quantization`: model-loading precision passed to MLLM scripts: `4b`, `8b`,
  or `16b`.
- `use_unsloth`: `true` selects the Unsloth training entrypoints; `false`
  selects the standard Transformers/TRL entrypoints.
- `eval_datasets_ft` and `eval_tasks_ft`: optional adapter-evaluation grid.
  Dataset names are the same as `datasets`; task names are the same as `tasks`.

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

`config/fundus_datasets.yaml` records the dataset registry and original-source download information used by the benchmark.

</details>

## Run Baseline Benchmarks

After benchmark data and models are available locally, submit the offline evaluation jobs:

```bash
sbatch jobs/eval/run_cv_benchmark.sh
sbatch jobs/eval/run_vlms_benchmark.sh
sbatch jobs/eval/run_mllms_benchmark.sh
```

Outputs are written under `$OUTPUT_DIR`, normally `results/evals`. The `.gitignore` excludes results because they are generated artifacts.

<details>
<summary><strong>Alternative A100 inference jobs</strong></summary>

The inference experiments reported in the paper use the H100 jobs above. Three
alternative A100 jobs are provided for environments with less capable GPU
hardware. They execute the same inference grids, read the same YAML
configurations, use the same evaluation parameters, and write to the same
`$OUTPUT_DIR`; only the requested GPU partition and architecture module differ.

```bash
sbatch jobs/eval/run_cv_benchmark_a100.sh
sbatch jobs/eval/run_vlms_benchmark_a100.sh
sbatch jobs/eval/run_mllms_benchmark_a100.sh
```

</details>

<details>
<summary><strong>Adapter training and evaluation (SFT / GRPO)</strong></summary>

### Train adapters

SFT and GRPO are driven by `config/mllm_config.yaml` using `models`, `train_datasets`, `train_tasks`, `sft_strategies`, `grpo_strategies`, `quantization`, and `use_unsloth`.

Run SFT (used in the paper):

```bash
sbatch jobs/train/run_sft.sh
```

Run GRPO (optional):

```bash
sbatch jobs/train/run_grpo.sh
```

Adapters are written under `$CHECKPOINTS_DIR`, normally:

```text
results/checkpoints/sft_<model>_<dataset>_<task>_<strategy>
results/checkpoints/grpo_<model>_<dataset>_<task>_<strategy>
```

These checkpoints are generated artifacts and should not be committed.

### Evaluate adapters

Evaluate trained SFT adapters:

```bash
sbatch jobs/eval/evaluate_sft_adapters.sh
```

Evaluate trained GRPO adapters:

```bash
sbatch jobs/eval/evaluate_grpo_adapters.sh
```

The adapter evaluation jobs skip missing checkpoint directories, so partial training grids are supported.

</details>

## Analyze Results

Aggregate baseline results:

```bash
sbatch jobs/eval/run_analysis.sh
```

<details>
<summary><strong>Analyze fine-tuned adapters</strong></summary>

Aggregate fine-tuned adapter results (SFT and any available GRPO outputs):

```bash
sbatch jobs/eval/run_analysis_ft.sh
```

For an SFT-only analysis, use:

```bash
sbatch jobs/eval/run_analysis_sft.sh
```

Analysis outputs are written to `$ANALYSIS_DIR` and `$FT_ANALYSIS_DIR`, defaulting to `results/analysis` and `results/analysis_ft`. Manuscript table/figure outputs are generated artifacts and are ignored by default.

</details>

<details>
<summary><strong>Direct Python entrypoints</strong></summary>

Set `PYTHONPATH` from the repository root when running scripts directly:

```bash
export PYTHONPATH="${PYTHONPATH:-}:."
```

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

</details>

<details>
<summary><strong>Programmatic API and embedding examples</strong></summary>

The benchmark modules can also be used directly when you want to build custom experiments, reuse the curated fundus loaders, or extract model embeddings outside the Slurm evaluation grid.

### Load A Dataset

`RetinaDataset` normalizes the supported fundus datasets into a common row interface. `base_dir` should be the same path used as `DATA_PATH` in `config/paths.env`.

```python
from retina_bench.core.data import RetinaDataset

dataset = RetinaDataset(
    base_dir="/path/to/prepared-fundus-datasets",
    dataset_name="brset",
    split="test",
)

print(len(dataset))
row = dataset.get_row(0)
image = dataset.get_image(0)

print(row[["image_id", "split", "DR_2_Class", "Task_Referable"]])
print(image.size)
```

### Extract Image Embeddings From CV Models

Image-only foundation models live under `retina_bench.cv`. They expose `get_image_embeddings(images)`.

```python
import torch

from retina_bench.core.data import RetinaDataset
from retina_bench.cv.models import get_cv_model

data = RetinaDataset("/path/to/prepared-fundus-datasets", "papila", split="test")
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

data = RetinaDataset("/path/to/prepared-fundus-datasets", "brset", split="test")
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

data = RetinaDataset("/path/to/prepared-fundus-datasets", dataset_name, split="test")
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

To inspect first-token answer probabilities, request scores for task labels when the underlying wrapper supports returning generation scores:

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

All model examples assume the required checkpoints are already present in the Hugging Face cache or under `ext_repos/`, because the benchmark is designed to run offline after the download stage. Use the evaluation jobs above to reproduce benchmark scores; the programmatic examples illustrate the model interfaces.

</details>

<details>
<summary><strong>Dataset access, model links, and additional notes</strong></summary>

- Dataset access and model terms must be accepted upstream when using original sources or gated checkpoints. Useful dataset source links:

  - BRSET: https://physionet.org/content/brazilian-ophthalmological/1.0.0/
  - mBRSET: https://physionet.org/content/mbrset/1.0/
  - PAPILA: https://figshare.com/articles/dataset/PAPILA/14798004
  - RFMiD: https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification
  - RFMiD 2.0: https://zenodo.org/records/7505822
  - IDRiD mirror used by the downloader: https://zenodo.org/records/17219542
  - Messidor-2 Kaggle mirror used by the downloader: https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess
  - G1020 Kaggle mirror used by the downloader: https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets
  - REFUGE official challenge: https://refuge.grand-challenge.org/
  - JSIEC1000 Kaggle mirror used by the downloader: https://www.kaggle.com/datasets/linchundan/fundusimage1000

  Useful Hugging Face model pages:

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

- Some model checkpoints are gated and require accepting upstream terms before `download_*` jobs can cache them.
- The benchmark expects benchmark-ready dataset folders under `$DATA_PATH` and cached models under `$HF_HOME`; evaluation and training jobs intentionally run offline.

</details>

# EyeCLIP: A Multimodal Visual–Language Foundation Model for Computational Ophthalmology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch\&logoColor=white)](https://pytorch.org/)

> Official repository for **EyeCLIP**, a vision-language foundation model designed specifically for **multi-modal ophthalmic image analysis**.
> 📝 **Paper**: [*A Multimodal Visual–Language Foundation Model for Computational Ophthalmology* (npj Digital Medicine, 2025)](https://www.nature.com/articles/s41746-025-01772-2.pdf)

---

## 🔍 Overview
**EyeCLIP** adapts the CLIP (Contrastive Language–Image Pretraining) architecture to address the unique challenges of ophthalmology. It incorporates self-supervised learning, multi-modal image contrastive learning, and hierarchical keyword-guided vision-language supervision. These innovations empower EyeCLIP to achieve **zero-shot disease recognition**, **cross-modal retrieval**, and **efficient fine-tuning** across a wide range of ophthalmic and systemic conditions.

---
## ✨ Key Features

* 🧠 **Multimodal Support**
  Natively pretrained on 11 ophthalmic modalities using one encoder, including:

  * Color Fundus Photography (CFP)
  * Optical Coherence Tomography (OCT)
  * Fundus Fluorescein Angiography (FFA)
  * Indocyanine Green Angiography (ICGA)
  * Fundus Autofluorescence (FAF)
  * Slit Lamp Photography
  * Ocular Ultrasound (OUS)
  * Specular Microscopy
  * External Eye Photography
  * Corneal Photography
  * RetCam Imaging

* 🔗 **CLIP-based Vision–Language Pretraining**
  Tailored adaptation of OpenAI’s CLIP for ophthalmic imaging and medical-language semantics.

* 🚀 **Zero-Shot Generalization**
  Classifies both **ophthalmic** and **systemic** diseases using natural language prompts—without task-specific fine-tuning.

* 🧩 **Versatile and Adaptable**
  Easily fine-tuned for downstream diagnostic tasks, including multi-label classification, systemic disease prediction, and rare disease diagnosis.

---

## 🗞️ News
* **2026-02**: Updated model weights. Fixed an issue where the text encoder weights were corrupted in the previous checkpoint. Please re-download the latest checkpoint to ensure proper functionality.
* **2025-07**: Initial release of pre-trained EyeCLIP model weights
* **2025-06**: Paper accepted by *npj Digital Medicine*
* **2025-03**: Public release of EyeCLIP codebase

---

## ⚙️ Installation

Set up the environment using conda and pip:

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
git clone https://github.com/Michi-3000/EyeCLIP.git
cd EyeCLIP
```

---

## 🎯 Pretrained Weights

| Model Name           | Description                                                    | Download Link                      |
| -------------------- | -------------------------------------------------------------- | ---------------------------------- |
| `eyeclip_visual` | Multimodal foundation model trained on diverse ophthalmic data | [🔗 Google Drive](https://drive.google.com/file/d/1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr/view?usp=sharing) |

---

## 📁 Dataset Preparation

To prepare datasets for pretraining or downstream evaluation:

1. **Download** datasets referenced in the paper.
2. **Organize** them into the following format:

```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels.csv
```

* `labels.csv` should follow the format:

```
impath,class
/path/to/image1.jpg,0
/path/to/image2.jpg,1
```

---

## 🚀 Quick Start

### 🔍 Zero-Shot Evaluation

```bash
python zero_shot.py \
    --data_path ./your_dataset \
    --text_prompts "normal retina,diabetic retinopathy,glaucoma"
```

---

### 🩺 Fine-Tuning

#### Ophthalmic Disease Classification

```bash
bash scripts/cls_ophthal.sh
```

Or use the Python version:

```bash
current_time=$(date +"%Y-%m-%d-%H%M")
FINETUNE_CHECKPOINT="eyeclip_visual.pt"
DATA_PATH="/data/public/"

DATASET_NAMES=("IDRiD" "Aptos2019" "Glaucoma_Fundus" "JSIEC" "Retina" "MESSIDOR2" "OCTID" "PAPILA")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    echo "=============================================="
    echo "Processing dataset: $DATASET_NAME"
    echo "=============================================="
    
    python main_finetune_ophthal.py \
        --data_path "${DATA_PATH}" \
        --data_name "${DATASET_NAME}" \
        --finetune "${FINETUNE_CHECKPOINT}" \
        --clip_model_type "ViT-B/32" \
        --batch_size 64 \
        --epochs 50 \
        --lr 1e-4 \
        --weight_decay 0.01 \
        --output_dir "./classification_results/${current_time}" \
        --warmup_epochs 5 \
        --test_num 5
        
    echo "Finished processing dataset: $DATASET_NAME"
    echo ""
done

echo "All datasets processed successfully!"
```

#### Systemic Disease Classification

```bash
bash scripts/cls_chro.sh
```

Or with custom parameters:

```bash
current_time=$(date +"%Y-%m-%d-%H%M")
FINETUNE_CHECKPOINT="eyeclip_visual.pt"

CUDA_VISIBLE_DEVICES=0 python main_finetune_chro.py \
    --finetune "${FINETUNE_CHECKPOINT}" \
    --clip_model_type "ViT-B/32" \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --output_dir "./classification_results/${current_time}" \
    --warmup_epochs 5 \
    --test_num 5
```

---

## 🧪 Pretraining from Scratch

To pretrain EyeCLIP on your own dataset:

```bash
python CLIP_ft_all_1enc_all.py
```

---

## 📚 Scripts and Utilities

| Script                     | Purpose                                              |
| -------------------------- | ---------------------------------------------------- |
| `main_finetune_ophthal.py` | Fine-tuning on ophthalmic disease datasets           |
| `main_finetune_chro.py`    | Fine-tuning for systemic (chronic) disease detection |
| `zero_shot.py`             | Zero-shot classification using language prompts      |
| `retrieval.py`             | Cross-modal image–text retrieval                     |

---

## 📖 Citation

If you use EyeCLIP in your research, please cite:

```bibtex
@article{shi2025multimodal,
  title={A multimodal visual--language foundation model for computational ophthalmology},
  author={Shi, Danli and Zhang, Weiyi and Yang, Jiancheng and Huang, Siyu and Chen, Xiaolan and Xu, Pusheng and Jin, Kai and Lin, Shan and Wei, Jin and Yusufu, Mayinuer and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={381},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

---

## 🤝 Acknowledgements

This project builds upon prior open-source contributions, especially:

* [CLIP](https://github.com/openai/CLIP) – Contrastive Language–Image Pretraining by OpenAI
* [MAE](https://github.com/facebookresearch/mae) – Masked Autoencoders by Facebook AI Research

We thank the open-source community and the medical imaging research ecosystem for their invaluable contributions.

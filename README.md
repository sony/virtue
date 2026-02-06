# VIRTUE: Visual-Interactive Text-Image Universal Embedder (ICLR 2026)

<a href="https://huggingface.co/Sony/xxx" target="_blank">
    <img alt="Homepage" src="https://img.shields.io/badge/🌐_Website-Project_Page-003366?logoColor=003366" style="height:22pt"/>
</a>
<a href="https://arxiv.org/abs/2510.00523" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-VIRTUE_Paper-003366?logo=arxiv&logoColor=003366" style="height:22pt"/>
</a>
<a target="_blank" href="https://github.com/sony/virtue">
    <img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github">
</a>
<a href="https://huggingface.co/Sony/VIRTUE-7B-SCaR" target="_blank">
    <img alt="VIRTUE-7B-SCaR" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-VIRTUE_Family-003366?color=ffc107&logoColor=white" style="height:22pt"/>
</a>
<a href="https://huggingface.co/datasets/Sony/SCaR-Eval" target="_blank">
    <img alt="SCaR Benchmark" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-SCaR_Benchmarks-003366?color=ffc107&logoColor=white" style="height:22pt"/>
</a>


This repo contains an official PyTorch implementation of [VIRTUE: Visual-Interactive Text-Image Universal Embedder](https://arxiv.org/abs/2510.00523) by Wei-Yao Wang, Kazuya Tateishi, Qiyu Wu, Shusuke Takahashi, Yuki Mitsufuji.

## Overview
Multimodal representation learning models have demonstrated successful operation across complex tasks, and the integration of vision-language models (VLMs) has further enabled embedding models with instruction-following capabilities.
However, existing embedding models lack visual-interactive capabilities to specify regions of interests from users (e.g., point, bounding box, mask), which have been explored in generative models to broaden their human-interactive applicability.
Equipping embedding models with visual interactions not only would unlock new applications with localized grounding of user intent, which remains unexplored, but also enable the models to learn entity-level information within images to complement their global representations for conventional embedding tasks.
In this paper, we propose a novel **V**isual-**I**nte**R**active **T**ext-image **U**niversal **E**mbedder (VIRTUE) that extends the capabilities of the segmentation model and the vision-language model to the realm of representation learning.
In VIRTUE, the segmentation model can process visual prompts that pinpoint specific regions within an image, thereby enabling the embedder to handle complex and ambiguous scenarios more precisely.
To evaluate the visual-interaction ability of VIRTUE, we introduce a large-scale **S**egmentation-and-Scene **C**}ption **R**etrieval (SCaR) benchmark comprising 1M samples that aims to retrieve the text caption by jointly considering the entity with a specific object and image scene.
VIRTUE consistently achieves a state-of-the-art performance with significant improvements across 36 universal MMEB (3.1\%–8.5\%) and five visual-interactive SCaR (15.2\%–20.3\%) tasks.

![Framework](./assets/teaser.png)

---

## How to Use VIRTUE
Please refer to the [example code](https://github.com/sony/virtue/blob/master/codes/virtue-example.py).
```
python3 virtue-example.py
```

---

## Folder Structure
.
├── virtue (this repo)
│   ├── assets
│   │   ├── example.jpg
│   │   └── teaser.png
│   ├── codes
│   │   ├── configs
│   │   ├── demo.py
│   │   ├── eval.py
│   │   ├── HF_model_conversion
│   │   ├── requirements.txt
│   │   ├── sam2_checkpoints
│   │   ├── scripts
│   │   ├── src
│   │   ├── train.py
│   │   └── virtue-example.py
│   ├── docker
│   │   └── virtue-env.Dockerfile
│   ├── docs
│   │   ├── index.html
│   │   └── static
│   └── README.md
├── data
│   ├── MMEB-train
│   ├── SCaR-eval
│   │   ├── images
│   │   │   ├── ade20k_val
│   │   │   ├── coco_stuff_val
│   │   │   ├── refcocog_val
│   │   │   ├── refcoco_plus_val
│   │   │   └── visualgenome_val
│   │   ├── SCaR_eval_ADE20K.parquet
│   │   ├── SCaR_eval_COCO_Stuff.parquet
│   │   ├── SCaR_eval_RefCOCOg.parquet
│   │   ├── SCaR_eval_RefCOCO_plus.parquet
│   │   └── SCaR_eval_VisualGenome.parquet
│   └── SCaR-train
│       ├── images
│       │   ├── ade20k_train
│       │   ├── coco_stuff_train
│       │   ├── refcocog_train
│       │   ├── refcoco_plus_train
│       │   └── visualgenome_train
│       ├── SCAR_ADE20K.parquet
│       ├── SCAR_COCO_Stuff.parquet
│       ├── SCAR_RefCOCOg.parquet
│       ├── SCAR_RefCOCO_plus.parquet
│       └── SCAR_VisualGenome.parquet
---

## Environment Setup
The environment is set with Python3.11.
```
conda create -n virtue python=3.11 -y
conda activate virtue
conda install pytorch=2.5 torchvision torchaudio pytorch-cuda=12.1 "mkl>=2023,<2025" -c pytorch -c nvidia -y
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && pip install flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r requirements.txt	        % in codes
pip install -e .		                % clone SAM2 and execute inside SAM2
```

We also provide the dockerfile under `docker/`.

## Data Preparation
### MMEB
Please download MMEB-train and MMEB-eval from [the official datasets](https://huggingface.co/collections/TIGER-Lab/vlm2vec), and set up the corresponding paths in `train_mmeb.yaml` and `eval_mmeb.yaml`.

### SCaR
Please specify the paths of the parquet files in `data_basedir` in the `virtual_eval.yaml`. Please download raw images from RefCOCO+, RefCOCOg, COCO-Stuff, VisualGenome, and ADE20K.

## Scripts
It is required to prepare [the sam2 checkpoints](https://github.com/facebookresearch/sam2) in advance and specify the corresponding path in:
```
sam_config: 
  config_path: "./sam2.1/sam2.1_hiera_b+.yaml"
  checkpoint: "/your/path/to/sam2_checkpoints/sam2.1_hiera_base_plus.pt"
```

If using the VIRTUE family, `model.py` will load the trained checkpoints for SAM.

### Train the model
Under the `codes` folder:
```
bash scripts/train_local.sh
```
For detailed hyper-parameters, please refer to the yaml files under `configs/`.

### Evaluate the model
Under the `codes` folder:
```
bash scripts/eval.sh
```
Change `dataset_config` in `virtue_eval.yaml` to `eval_scar.yaml` or `eval_mmeb.yaml` for SCaR or MMEB.

### Local Demonstration
Under the `codes` folder:
```
python3 demo.py
```
Please change `MSCOCO_IMAGES_DIR` to the corresponding image folder.

### Convert Pytorch models into Huggingface ones
1. `convert_hf_model.py` converts the trained PyTorch model to the HuggingFace format based on the model path in `virtue_eval.yaml`.
```
python3 convert_hf_model.py
```
> You may need to copy some configuration files to the HF folder.
2. `load_hf_model.py` provides a quick example use to see if it works.
```
python3 HF_model_conversion/load_hf_model.py
```

## Some Known Issues
- According to the authors of VLM2Vec_v2, the trainig codebase cannot naturally work for single-GPU, which will cause errors for GradCache.

## Contact
For any questions or issues, please feel free to open an issue/PR or reach out to Wei-Yao Wang.

## Citation
If you found this repository is relevant or useful to your research, please consider citing our paper:
```
@article{wangICLR2026virtue,
  author       = {Wei-Yao Wang and
                  Kazuya Tateishi and
                  Qiyu Wu and
                  Shusuke Takahashi and
                  Yuki Mitsufuji},
  title        = {VIRTUE: Visual-Interactive Text-Image Universal Embedder},
  journal      = {arXiv preprint arXiv:2510.00523},
  year         = {2025}
}
```
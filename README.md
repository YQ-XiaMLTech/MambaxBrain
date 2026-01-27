# Introduction to MambaxBrain

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of **"Tau-BNO"**.

> **Abstract:** Functional magnetic resonance imaging (fMRI) contains rich individual, cognitive, and pathological information, but lacks a universal model capable of multi-task modeling of these multidimensional aspects of brain function. Here, we introduce MAMBAxBrain, a multi-task neural framework that integrates the efficient Mamba architecture with functional connectivity (FC) analysis to jointly model the temporal dynamics and spatial coordination of neural activity.
By combining these complementary features, MAMBAxBrain achieves accurate prediction and identification across diverse fMRI tasks, including brain fingerprinting, cognitive task decoding, reaction time prediction, and schizophrenia diagnosis.
Across four representative tasks, the framework consistently outperformed state-of-the-art benchmark methods and exhibited robust generalization to cross-session data.
Multi-task learning experiments further demonstrated its generalization across individual, cognitive, and behavioral dimensions. 
Interpretability analyses reveals distinct neural circuits across tasks: higher-order default mode and control networks for individual identity, sensory–motor systems for cognitive states, and a subcortical–motor–limbic pathway for reaction time. For schizophrenia, the model identified impairments in cross-modal sensory integration (auditory-visual) and the top-down regulation of these streams by frontoparietal and cingulo-opercular control hubs. These findings show that our model uncovers meaningful, task-specific neural signatures.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🔬 Introduction

This repository contains the official PyTorch implementation of MAMBAxBrain, a multi-task framework for modeling multiregional fMRI time series. fMRI signals capture rich information about individual identity, cognitive states, and neuropsychiatric conditions, making them a valuable modality for large-scale brain modeling.

MAMBAxBrain combines Mamba-based temporal modeling with functional connectivity representations to capture both within-region dynamics and cross-region interactions. This unified design enables strong performance across diverse downstream tasks, including brain fingerprinting, task decoding, reaction-time regression, and schizophrenia classification.

**MAMBAxBrain Architecture**

![image-20251121143349619](images/structure.png)

---

## ✨ Key Features

- **MAMBAxBrain Architecture**: A multi-task architecture that integrates Mamba-based temporal modeling with FC representations via a dual-stream design and feature fusion.
- **Comprehensive Evaluation**: Extensive testing and visualization tools
- **Reproducibility**: Complete training and evaluation scripts with detailed configurations
- **Multiple Baselines**: Implementations and comparisons with BrainNetTF, Mamba, FSTMamba, SPDNet

---

## 📁 Repository Structure

```
MambaxBrain/
├── dataset/                   # Dataset directory
├── mamba_ssm/                 # Mamba/SSM backbone modules
├── model/                     # MAMBAxBrain model definitions
├── result_model/              # Saved checkpoints / pre-trained weights
├── config.py                  # Training configurations
├── figure.py                  # Visualization scripts
├── main.py                    # Main entry for training (FP, cognitive TD, RT, and multi-task; excludes SZ)
├── main_sz.py                 # Main entry for training (Schizophrenia diagnosis only)
├── test_model.py              # Main entry for evaluation
├── test_task.py               # Testing script
├── train_3tasks.sh            # Shell script: joint training for three tasks
├── train_cognitive.sh         # Shell script: cognitive task decoding training
├── train_FP.sh                # Shell script: brain fingerprinting training
├── train_RT.sh                # Shell script: reaction-time regression training
├── train_sz.sh                # Shell script: Schizophrenia diagnosis training
├── utils.py                   # Common utilities
└── requirements.txt           # Python dependencies
```

---

## 🚀 Getting Started

### Requirements

- Python 3.8.10
- PyTorch 2.0.0 (CUDA 11.8 build: `2.0.0+cu118`)
- CUDA 11.8 (recommended for GPU training)
- Other dependencies listed in `requirements.txt`
### Installation

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd MambaxBrain
   ```

2. **Create and activate conda environment**

   ```bash
   conda create -n mambaxbrain python=3.8.10 -y
   conda activate mambaxbrain
   ```

3. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```
---

### Dataset Preparation

This project expects **preprocessed ROI time-series** stored as `.npy` files.

#### 1) Organize the preprocessed time-series

Place your preprocessed data under `processed_timeseries_114x1/` (default). The expected structure is:

```
processed_timeseries_114x1/
├── Rest/
│   └── natural/
│       ├── subj0001_t0.npy
│       ├── subj0001_t1.npy
│       └── ...
├── WM/
│   └── natural/
│       ├── subj0001_t0.npy
│       └── ...
├── LANGUAGE/
│   └── natural/
├── MOTOR/
│   └── natural/
├── SOCIAL/
│   └── natural/
├── GAMBLING/
│   └── natural/
├── RELATIONAL/
│   └── natural/
└── EMOTION/
    └── natural/
```

**File format**

* Each `.npy` file is a short ROI time-series segment.
* For HCP-style tasks, each segment is reshaped to **(114, 1)** and concatenated along time.
* For `Rest`, the first **2400** time points are used for training/testing, and the remaining part is used as a **retest split**.

> **Tip:** You can change the root directory via `data_dir` in the loader functions.

---

#### 2) Subject mapping files (required)

Place the following CSV files under `dataset/`:

```bash
mkdir -p dataset
# Expected mapping files:
# - dataset/subid_mapping.csv
# - dataset/Schizophrenia_mapping.csv
# - dataset/story_mask.csv
# - dataset/math_mask.csv
# - dataset/TOM_mask.csv
# - dataset/RC_mask.csv
# - dataset/twoback_mask.csv
```

**(a) `dataset/subid_mapping.csv`**
Used for **fingerprinting / task decoding / reaction-time regression labels**.

Required columns:

* `sub_idx`: subject numeric index (parsed from `subjXXXX`)
* `Label`: subject ID label for fingerprinting (use `-1` to exclude subjects)
* Reaction-time columns (used in regression):

  * `STORY_RT`, `MATH_RT`, `Relational_RT`, `TOM_RT`, `Random_RT`, `TwoBackRT`

**(b) Mask files for sub-task extraction (regression)**
* `dataset/story_mask.csv`
* `dataset/math_mask.csv`
* `dataset/TOM_mask.csv`
* `dataset/RC_mask.csv` (Random/RC mask)
* `dataset/twoback_mask.csv`

Each mask is a 0/1 vector (CSV) indicating which time points to keep.

**(c) `dataset/Schizophrenia_mapping.csv`**
Used for SZ classification.

Required columns:

* `subID`: subject ID string (matches filename prefix, e.g., `subjXXXX`)
* `diagnosis`: one of `{Healthy, HighRisk, SZ}` (Healthy & HighRisk are mapped to class 0, SZ to class 1)
* `converter`: subjects with `converter == 1` will be excluded

---

## 💻 Usage

### Training

We provide training scripts for different tasks:

* `train_FP.sh`: Brain fingerprinting (subject identification)
* `train_cognitive.sh`: Cognitive task decoding
* `train_RT.sh`: Reaction-time regression
* `train_3tasks.sh`: Joint training on three tasks (multi-task)

Below are example commands.

#### 1) Brain Fingerprinting (FP)

```bash
bash train_FP.sh
```

#### 2) Cognitive Task Decoding

```bash
bash train_cognitive.sh
```

#### 3) Reaction-Time Regression (RT)

```bash
bash train_RT.sh
```

#### 4) Multi-task Training (3 tasks)

```bash
bash train_3tasks.sh
```

---

### Common Training Parameters (in `config.py`)

Most training hyperparameters and paths are defined in `config.py`. Typical options include:

* `device`: GPU device ID (e.g., `0`)
* `seed`: Random seed for reproducibility
* `data_dir`: Path to preprocessed ROI time-series (default: `processed_timeseries_114x1`)
* `condition`: Condition name under each task directory (default: `natural`)
* `train_ratio`: Train/test split ratio within each subject’s time series (default: `0.7`)
* `epochs`: Number of training epochs
* `batch_size`: Batch size
* `lr`: Learning rate
* `weight_decay`: L2 regularization coefficient
* `num_rois`: Number of ROIs (e.g., `114` for HCP tasks; `86` for SZ)
* `task`: Training task mode (`FP`, `cognitive`, `RT`, or multi-task)

> **Note:** If you modify dataset paths or task settings, please update them in `config.py` accordingly.

---
### Evaluation

We provide a unified evaluation script for the 3-task setting:

```bash
bash test_3tasks.sh
```

This script evaluates a pretrained checkpoint from `result_model/3task` on three tasks (**fingerprinting**, **cognitive decoding**, and **reaction-time regression**) using:

* `--modes`: `{"fingerprint":"test","cognitive":"test","rt":"test"}`
* `--run_dir`: `result_model/3task`
* `window_size=1`, `sequence_length=100`, `batch_size=64`
* `train_ratio=0.7`, `seed=123`, `condition="natural"`
* `--gpu` (GPU inference enabled)

---

## 📊 Results

### Quantitative Results

Performance comparison across methods for **in-domain evaluation (HCP-YA)** and **cross-session generalization (HCP-RT)**, as well as **SZ diagnosis on UCSF**.
Accuracy is reported for classification tasks: **brain fingerprinting (FP)**, **cognitive task decoding (TD)**, and **schizophrenia diagnosis (SZ)**.
For the regression task **reaction time prediction (RT)**, we report the **(R^2 \times 100%)** score. All results are reported as **mean ± standard deviation**.

| **Model**              | **UCSF: SZ Diagnosis (Acc ↑)** | **HCP-YA: Brain FP (Acc ↑)** | **HCP-YA: Cognitive TD (Acc ↑)** | **HCP-YA: RT Prediction (R²×100% ↑)** | **HCP-RT: Brain FP (Acc ↑)** | **HCP-RT: Cognitive TD (Acc ↑)** |
| ---------------------- | ------------------------------ | ---------------------------- | -------------------------------- | ------------------------------------- | ---------------------------- | -------------------------------- |
| SPDNet                 | 79.63 ± 0.69                   | 70.91 ± 0.02                 | 95.25 ± 0.08                     | 82.24 ± 0.96                          | 69.74 ± 0.23                 | 93.37 ± 0.41                     |
| BrainNetTF             | 84.79 ± 0.53                   | 74.55 ± 3.04                 | 98.75 ± 0.07                     | 88.29 ± 1.04                          | 73.37 ± 2.61                 | 96.81 ± 0.05                     |
| Mamba                  | 82.36 ± 0.63                   | 94.95 ± 0.50                 | 99.43 ± 0.12                     | 88.10 ± 0.71                          | 83.26 ± 0.82                 | 97.41 ± 0.34                     |
| FSTMamba               | 85.46 ± 0.91                   | 94.34 ± 0.14                 | 98.82 ± 0.40                     | 89.01 ± 0.04                          | 87.19 ± 0.62                 | 96.12 ± 0.09                     |
| **MAMBAxBrain (Ours)** | **91.05 ± 0.71**               | **96.07 ± 0.55**             | **99.47 ± 0.06**                 | **91.80 ± 0.40**                      | **94.94 ± 0.49**             | **98.93 ± 0.38**                 |

**Key Findings:**

* **MAMBAxBrain achieves the best performance across all tasks and settings**, including SZ diagnosis (UCSF), in-domain evaluation (HCP-YA), and cross-session generalization (HCP-RT).
* On **HCP-RT cross-session generalization**, MAMBAxBrain improves over the strongest baseline (FSTMamba) by:

  * **+7.75** points on Brain FP (94.94 vs 87.19)
  * **+2.81** points on Cognitive TD (98.93 vs 96.12)
* On **UCSF SZ diagnosis**, MAMBAxBrain exceeds the best baseline (FSTMamba) by **+5.59** points (91.05 vs 85.46).
* On **RT prediction (HCP-YA)**, MAMBAxBrain achieves the highest (R^2\times100%) score (**91.80**), outperforming the best baseline (FSTMamba, 89.01) by **+2.79** points.


### Qualitative Results

**Error heatmaps showing prediction accuracy across brain regions**

![image-20251121143156979](./assets/image-20251121143156979.png)

**Trajectory accross ten different coefficient conditions**

![image-20251121143310994](./assets/image-20251121143310994.png)



---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex

```

---

## 📄 License

---

## 📧 Contact

For questions or issues, please:

- Open an issue in this repository
- Contact: [Yuqing (gmail)](xiayuqing0526@gmail.com)

---

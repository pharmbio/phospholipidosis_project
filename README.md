# Phospholipidosis Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A codebase for training and evaluating of predicting phospholipidosis readout from Cell Painting data and molecular descriptors:
- **CellProfiler** & **DeepProfiler** morphological feature pipelines  
- **Compound** embedding & classification workflows  
- **Multimodal** fusion of image + chemical data  
- **Conformal prediction** for uncertainty estimation  

This project is under development.

---

## Repository Layout
 
**General scripts**  
- `classification.py` — train single‐cell classifiers  
- `generate_crops.py` — extract and load image crops for CP pipeline  
- `run_training.nf` — Nextflow wrapper for classification

**/CP** – CellProfiler  
- `classification.py`  
- `generate_crops.py`  
- `Grit_check.ipynb`  
- `run_training.nf`  

**/Compounds** – Chemical embeddings & models  
- `compound_classification.py`  
- `compounds_embedding.ipynb`  

**/DP** – DeepProfiler  
- `DP_exploration.ipynb`  
- `DP_exploration_conformal.ipynb`  
- `DP_exploration_conformal_site.ipynb`  
- `DP_regression.ipynb`  

**/multimodal** – Image + compound fusion  
- `dataset.py`  
- `model.py`, `model_v2.py`  
- `train.py`  
- `optuna_tuning.py` & `sweep_config.yaml`  
- `wandb_sweep.py`  
- `utils.py`  

---

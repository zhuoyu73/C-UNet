# ArtifactNet: Complex-Valued Artifact Removal for MRE

This repository provides a full pipeline for preprocessing, training, and evaluating a deep learning model (`ArtifactNet`) designed for artifact removal in complex-valued Magnetic Resonance Elastography (MRE) images.

---

## 📁 Directory Structure

```
.
├── preprocess_artifactnet.py
├── src/
│   └── models/artifactnet.py
│   └── visualize_artifacts.py
│   └── pipeline.py
│   └── datasets.py
├── data/
│   └── v0/
│       ├── training.txt
│       ├── validation.txt
│       └── test.txt
├── data_processed_artifact_removal/
│   ├── training/
│   ├── validation/
│   └── test/
├── runs_artifactnet/
│   └── <timestamp>/
│       ├── best_model.pt
│       ├── metrics.csv
│       └── (tensorboard logs)
└── ...
```

---

## 🔧 Requirements

Ensure the following Python packages are installed:

```bash
pip install torch torchvision scipy numpy pandas matplotlib tqdm
```

Make sure PyTorch is installed with GPU (CUDA) support for optimal performance.

---

## 💾 Step 1: Preprocessing

Run the preprocessing script to convert `.mat` files into `.pt` tensors:

```bash
python preprocess_artifactnet.py
```

### Input Format

- Root directory: `/mnt/external/zhuoyu/fully+osci/`
- For each subject:
  - Fully sampled: `<subject_id>/<subject_id>_B1000/mbmre/img.mat`
  - Low-rank: `<subject_id>/<subject_id>_B1000/mbmre_both/img_US.mat`
- Subject IDs should be listed in:
  - `data/v0/training.txt`
  - `data/v0/validation.txt`
  - `data/v0/test.txt`

### Output

Processed tensors are saved to `data_processed_artifact_removal/{split}/`.

---

## 🧠 Step 2: Training

To train the model:

```bash
python -m src/pipeline
```

This will automatically:
- Load preprocessed data
- Train the ArtifactNet model for 50 epochs
- Save checkpoints and logs under `runs_artifactnet/<timestamp>/`

Add `--debug` to quickly test the training loop with fewer steps.

---

## 👁️ Step 3: Visualization and Inference

Run inference and visualize predictions:

```bash
python visualize_artifacts.py \
    --clean /path/to/clean/img.mat \
    --lowrank /path/to/lowrank/img.mat \
    --model runs_artifactnet/<timestamp>/best_model.pt \
    --save ./results
```

This will:
- Identify the slice with minimum MSE
- Visualize real and imaginary components of clean, lowrank, predicted artifact, and denoised output
- Save outputs: `artifact_pred.mat`, `denoised_img.mat`, `denoised_img_3.mat`

---

## 📊 Evaluation Metrics

- MSE between predicted and true artifact
- PSNR (magnitude & phase) for artifact and denoised outputs

---

## 🔍 Notes

- Each input slice is in `[2, 120, 120]` (real/imag), zero-padded to `[2, 128, 128]`
- All values are scaled by `1e6` before processing
- The model operates on complex-valued images using real and imaginary channels
- Optional circular brain mask weighting available in the loss function

---

## 👩‍💻 Author

Zhuoyu Shi  
Biomedical Engineering PhD Student  
Columbia University, 2025

This document was generated with the assitance of GPT-4o. 

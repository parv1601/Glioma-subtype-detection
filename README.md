# Glioma Subtype Classification & Analysis Pipeline

This project implements a robust, weakly supervised machine learning pipeline for the classification of Glioma subtypes (Glioblastoma, Astrocytoma, and Oligodendroglioma) using Whole Slide Images (WSIs). It features two distinct foundation model architectures: **CHIEF** and **IPD-Brain**, both utilizing the **DTFD-MIL** (Double-Tier Feature Distillation Multiple Instance Learning) aggregator for final WSI-level classification.

## 🌟 Overview
The pipeline transforms high-resolution gigapixel slides into diagnostic predictions through a systematic four-step process:
1.  **Patch Extraction:** Segmenting the tissue and tiling the WSI into smaller patches.
2.  **Feature Extraction:** Encoding patches into low-dimensional embeddings using foundation models.
3.  **DTFD Aggregator Training:** Training a MIL model to aggregate patch features into a slide-level prediction.
4.  **Inference & Visualization:** Generating predictions on new data and visualizing diagnostic Regions of Interest (ROI) via attention heatmaps.

---

## 🏗️ Pipelines Implemented

### 1. CHIEF Pipeline
Based on the **Clinical Histopathology Imaging Evaluation Foundation (CHIEF)** model. It uses a transformer-based encoder (CTransPath) to extract highly generalizable pathology representations.
*   **Feature Dimension:** 768
*   **Foundation Model:** CTransPath

### 2. IPD-Brain Pipeline
A specialized pipeline optimized for brain pathology, utilizing a **ResNet-50** backbone pretrained with specialized clinical datasets.
*   **Feature Dimension:** 2048
*   **Foundation Model:** ResNet-50 (Custom clinical weights)

---

## 📥 Step 0: Data Downloading
You can download the necessary Glioma slides from the [GDC Portal](https://portal.gdc.cancer.gov/) using the provided manifest files.

```bash
# Install GDC client if not already available
# Download slides using the provided manifests
gdc-client download -m gdc_manifest.txt
gdc-client download -m gdc-manifest-set2.txt
```
*Ensure all `.svs` files are placed in a directory named `glioma_slides` or as configured in the scripts.*

---

## 🔪 Step 1: Patch Extraction
Segment tissue and create patches for both pipelines.

### Execution
Run the following command within the respective pipeline folder (`CHIEF/` or `IPD-Brain/`):

```bash
python create_patches_fp.py \
    --source /path/to/glioma_slides \
    --save_dir ../processed_data/patches \
    --patch_size 224 \
    --preset ipd_preset.csv \
    --seg --patch --stitch
```
*   **CHIEF location:** `CHIEF/create_patches_fp.py`
*   **IPD location:** `IPD-Brain/create_patches_fp.py`
*   **Output:** Generates `.h5` files containing patch coordinates and `.png` masks/stitches.

---

## 🧬 Step 2: Feature Extraction
Convert patches into numerical feature vectors.

### IPD-Brain (ResNet-50 Features)
```bash
python IPD-Brain/extract_features.py
```
*   **Input:** Patches from `../processed_pilot/patches`
*   **Output:** Saved as `.pt` files in `../processed_pilot/features/pt_files`

### CHIEF (CTransPath Features)
```bash
python CHIEF/Get_CHIEF_patch_feature.py
```
*   **Input:** Patches from `../processed_chief/patches`
*   **Output:** Saved as `.pt` files in `../processed_chief/features/pt_files`

---

## 🚂 Step 3: DTFD Aggregator Training
Train the Multiple Instance Learning (MIL) model using the extracted features.

### For IPD-Brain
```bash
python IPD-Brain/Main_DTFD_MIL.py \
    --dataset_csv ../slides_labels_final.csv \
    --data_dir ../processed_pilot/features \
    --num_cls 3 \
    --in_chn 2048 \
    --mDim 384
```

### For CHIEF
```bash
# Or run the provided bash script for optimized parameters
bash CHIEF/train_v6_final.sh
```
*The `train_v6_final.sh` script includes optimized hyperparameters: Learning Rate (5e-5), Dropout (0.3), and Weight Decay (0.001).*

---

## 📊 Step 4: Inference & Evaluation

### Running Inference
To run inference on new slides using the trained IPD-Brain model:
```bash
python IPD-Brain/inference_script.py
```

### Confusion Matrix Generation
Generate performance metrics and confusion matrices for the test set:
```bash
# For CHIEF
python CHIEF/confusion.py

# For IPD-Brain
python IPD-Brain/confusion.py
```
This will output the **Confusion Matrix** and a detailed **Classification Report** (Precision, Recall, F1-score).

---

## 🗺️ Interactive Heatmap Viewer
The interactive viewer allows you to visualize the Regions of Interest (ROI) that the model focused on to make its prediction. It highlights the most "important" patches based on the attention weights from the MIL model.

### Key Features
- **Dual Pipeline Support:** Supports both CHIEF and IPD-Brain Focal models.
- **Dynamic Thresholding:** Adjust the attention threshold percentile in real-time.
- **Visualization Modes:** Toggle between "Boundaries" (clean ROI outlines) and "Heatmap" (weighted color overlays).
- **Metadata Integration:** Displays both predicted and actual labels for comparison.
- **Export:** Download the generated ROI visualizations as PNG files.

### How to Run
```bash
python launch_viewer.py
```
This script will:
1. Activate the virtual environment.
2. Install/Verify `streamlit`.
3. Launch the web-based interactive viewer (`interactive_heatmap_viewer.py`) at `http://localhost:8501`.

### Visualization Example
![ROI Heatmap Example](images/heatmap_example.png)
*The heatmap indicates the possible ROI of the inference, providing explainability for the diagnostic decision.*

---

## 🛠️ Project Structure
```text
.
├── CHIEF/                      # CHIEF Pipeline scripts
│   ├── Main_DTFD_MIL.py        # DTFD Training
│   ├── Get_CHIEF_patch_feature.py # Feature extraction
│   └── confusion.py            # Evaluation
├── IPD-Brain/                  # IPD-Brain Pipeline scripts
│   ├── Main_DTFD_MIL.py        # DTFD Training
│   ├── extract_features.py     # Feature extraction
│   └── confusion.py            # Evaluation
├── launch_viewer.py            # Heatmap viewer entry point
├── slides_labels_final.csv     # Master label file
└── requirements.txt            # Project dependencies
```

## 📜 Requirements
Install all dependencies via pip:
```bash
pip install -r requirements.txt
```
*Key dependencies: PyTorch, OpenSlide, Scikit-learn, Streamlit, Pandas, H5py.*

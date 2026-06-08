# Bread Quality Classification using CNN

**Machine Learning Project – Phase II** · Idea 34 (Bakery Product Quality Classification)

A Convolutional Neural Network that classifies **bread** as **Good Quality** vs
**Defective** (stale, burnt, undercooked or moldy), benchmarked against two
transfer-learning models (**MobileNetV2** and **ResNet50**). Built as a
decision-support / screening tool for bakery quality inspection.

---

## 📋 Project Overview

| | |
|---|---|
| **Task** | Binary image classification (good vs defective bread) |
| **Models** | Custom CNN + MobileNetV2 + ResNet50 |
| **Framework** | TensorFlow / Keras |
| **Environment** | Kaggle Notebook (GPU + Internet) |
| **Dataset** | ~500+ images, ≈250 good / ≈250 defective |

---

## 🗂️ Dataset

**"Bread" – Good and Bad Classification of Bread (Triticum Aestivum)**
Das & Sarkar, Mendeley Data — DOI **10.17632/2cymbb4gt4.1** — Licence **CC BY 4.0**
https://data.mendeley.com/datasets/2cymbb4gt4/1

| Class | Description |
|---|---|
| `good` | Fresh, high-quality bread (even golden crust, uniform texture) |
| `defective` | Stale, burnt, undercooked, moldy or discolored bread |

The notebook **downloads the dataset automatically** (Step 0) — no manual upload.

---

## 📂 Repository Structure

```
.
├── bakery_quality_classification.ipynb   # Main notebook (downloads data + full pipeline)
├── bakery_quality_classification.py       # Script version of the notebook
├── Proposal_Bakery_Product_Quality_Classification.docx   # Project proposal
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── output/                                # Output figures (generated after running)
```

---

## ▶️ How to Run (Kaggle)

1. Create a **New Notebook** on https://www.kaggle.com and import
   `bakery_quality_classification.ipynb` (File → Import Notebook).
2. **Settings → Accelerator → GPU.**
3. **Settings → Internet → On** (required — Step 0 downloads the dataset, and the
   transfer models download ImageNet weights). Internet needs a phone-verified account.
4. **Run All.** Step 0 pulls the ~4 GB dataset directly from Mendeley and extracts it;
   no "Add Input" or manual upload needed.

Outputs are saved to `/kaggle/working/figures/` and `/kaggle/working/models/`.

---

## 🔬 Pipeline

Automatic dataset download → cleaning (remove corrupted/duplicate images) +
uniqueness audit → preprocessing → resize + normalize → stratified
train/val/test split → data augmentation → custom CNN → two transfer models
(MobileNetV2, ResNet50) → training → evaluation (accuracy, precision, recall, F1,
confusion matrix, ROC-AUC, inference time) → model comparison → Grad-CAM
(correct / false-positive / false-negative) → error analysis → deployment & ethics
discussion → saved figures and models.

---

## 📊 Outputs

- Class distribution chart
- Sample and augmented image grids
- Accuracy / loss curves (3 models)
- Confusion matrices and classification reports (3 models)
- Overlaid ROC curves with AUC (all models)
- Model comparison table + bar chart (accuracy / F1 / AUC)
- Inference-time comparison chart
- Grad-CAM overlays for correct / false-positive / false-negative cases
- Misclassified-image error analysis

---

## 🛠️ Tech Stack

TensorFlow / Keras · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · Pillow

## 📚 Citation

Das, Ranjika; Sarkar, Tanmay (2024), *"Bread"*, Mendeley Data, V1,
doi: 10.17632/2cymbb4gt4.1 (CC BY 4.0).

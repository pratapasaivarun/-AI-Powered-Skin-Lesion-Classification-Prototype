# 🔬 AI-Powered Skin Lesion Classification Prototype

An educational research prototype for dermoscopic skin lesion classification using the **HAM10000** dataset, with **Grad-CAM** visualizations and an interactive **Streamlit** web application. It is not for medical use.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Download Dataset](#1-download-dataset)
  - [Train a Model](#2-train-a-model)
  - [Evaluate](#3-evaluate)
  - [Predict a Single Image](#4-predict-a-single-image)
  - [Run Web App](#5-run-the-web-app)
  - [Google Colab](#6-google-colab)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Workflow Diagram](#-workflow-diagram)
- [Configuration](#-configuration)
- [Technologies Used](#-technologies-used)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🎯 Overview

This project explores **Convolutional Neural Networks (CNNs)** and **Transfer Learning (MobileNetV2)** to classify 7 types of skin lesions from dermoscopic images, providing:

- **Automated classification** with confidence scores
- **Risk-category guidance** (🟢 GREEN / 🟡 YELLOW / 🔴 RED)
- **Grad-CAM heatmaps** showing which regions influenced the AI's decision
- **Cautious next-step guidance** based on predicted class and model uncertainty
- **Photo-quality prompts** for small, blurry, or unusually lit uploads

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **3 Model Architectures** | Custom CNN, MobileNetV2 (transfer learning), Hybrid CNN+MobileNetV2 |
| **Memory-Efficient Pipeline** | Uses `ImageDataGenerator.flow_from_dataframe()` — no RAM-heavy image arrays |
| **Two-Stage Training** | Stage 1: frozen backbone → Stage 2: fine-tuned top layers |
| **Class Imbalance Handling** | Automated balanced class weights + data augmentation |
| **Explainable AI (XAI)** | Grad-CAM heatmaps for model interpretability |
| **Confidence-aware guidance** | Low-confidence results receive a clinician-review prompt |
| **Photo-quality prompts** | Non-blocking warnings for small, blurry, or poorly lit uploads |
| **Streamlit Web App** | Interactive upload → classify → explain → recommend |
| **CLI Interface** | Train, evaluate, predict from the command line |
| **Google Colab Ready** | Jupyter notebooks for Colab with GPU acceleration |

---

## 📁 Project Structure

```
Skin cancer/
├── main.py                          # CLI entry point (train / evaluate / predict / download)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
│
├── src/                             # Core source code
│   ├── __init__.py
│   ├── config.py                    # All hyperparameters, paths, class metadata
│   ├── data_loader.py               # HAM10000 loading, splitting, generators
│   ├── models.py                    # 3 model architectures (Custom CNN, MobileNetV2, Hybrid)
│   ├── train.py                     # Two-stage training pipeline
│   ├── evaluate.py                  # Metrics, confusion matrix, training curves
│   ├── gradcam.py                   # Grad-CAM explainability
│   └── predict.py                   # Single-image prediction + next-step guidance
│
├── app/                             # Streamlit web application
│   └── streamlit_app.py
│
├── notebooks/                       # Jupyter / Colab notebooks
│   ├── skin-cancer.ipynb                             # Original Kaggle notebook (28×28, Custom CNN)
│   ├── Skin_Cancer_CustomCNN_v3.ipynb                # Custom CNN (224×224, Grad-CAM)
│   ├── Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb   # Hybrid CNN+MobileNetV2
│   └── AI Skin Cancer Detection Notebook.md          # Design specification
│
├── dataset/                         # HAM10000 dataset (downloaded via CLI)
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
├── models/                          # Saved trained models
│   ├── skin_cancer_best.keras
│   ├── skin_cancer_model.keras
│   └── label_encoder.pkl
│
└── results/                         # Evaluation outputs
    ├── training_curves.png
    ├── confusion_matrix.png
    └── training_history.pkl
```

---

## 📊 Dataset

**HAM10000** (*Human Against Machine with 10000 training images*) — a large benchmark dataset of dermoscopic images.

The included `hmnist_28_28_RGB.csv` contains 10,015 28×28 RGB images across
seven classes and is used by `run_full_pipeline.py`. Its train, validation, and
test split is stratified by class at the image level. The CSV does not include
lesion IDs, so images of the same lesion could appear in more than one split;
this is a potential source of data leakage and should be disclosed when
discussing evaluation. The separate 224×224 pipeline expects the original
metadata CSV and image folders; when lesion IDs are available, it groups those
IDs across splits. Generated model artifacts are local and ignored by Git, so a
fresh checkout must train a model before inference.

| Property | Value |
|----------|-------|
| Total images | 10,015 |
| Image size | 600×450 (resized to 224×224) |
| Classes | 7 |
| Source | [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |

### Lesion Classes

| Code | Full Name | Risk Level | Description |
|------|-----------|------------|-------------|
| `akiec` | Actinic Keratoses | 🔴 RED | Precancerous — may progress to squamous cell carcinoma |
| `bcc` | Basal Cell Carcinoma | 🔴 RED | Most common skin cancer; slow-growing but locally invasive |
| `mel` | Melanoma | 🔴 RED | Most dangerous skin cancer; can metastasize |
| `bkl` | Benign Keratosis | 🟡 YELLOW | Non-cancerous growths (seborrheic keratoses, solar lentigo) |
| `df` | Dermatofibroma | 🟡 YELLOW | Benign fibrous skin nodules |
| `vasc` | Vascular Lesions | 🟡 YELLOW | Blood vessel-related (angiomas, angiokeratomas) |
| `nv` | Melanocytic Nevi | 🟢 GREEN | Common benign moles |

---

## 🧠 Model Architectures

### 1. Custom CNN (`custom_cnn`)
A deep CNN trained from scratch with 4 convolutional blocks:
```
Conv2D(32) → BN → Pool → Conv2D(64)×2 → BN → Pool → Conv2D(128)×2 → BN → Pool →
Conv2D(256)×2 → BN → Pool → GAP → Dense(256→128→64→32→7)
```

### 2. MobileNetV2 Transfer Learning (`mobilenet_v2`)
ImageNet-pretrained MobileNetV2 backbone with a custom classification head:
```
MobileNetV2(frozen) → GAP → BN → Dense(512) → BN → Dense(256) → BN → Dense(128) → Dense(7)
```
Fine-tuning: top 50 layers unfrozen in Stage 2.

### 3. Hybrid CNN + MobileNetV2 (`hybrid`)
Dual-branch architecture combining both approaches:
```
Branch A: Custom CNN → GAP → Dense(128)  ─┐
                                           ├─ Concatenate → Dense(256→128→64→7)
Branch B: MobileNetV2 → GAP → Dense(128) ─┘
```
Fine-tuning: top 30 MobileNetV2 layers unfrozen in Stage 2.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone or navigate to the project
cd "c:\PROJECTS\Skin cancer"

# Install dependencies
pip install -r requirements.txt

# Start the app (works before training; analysis becomes available once artifacts exist)
streamlit run app/streamlit_app.py
```

Training is optional and is never started by launching the app. To enable
inference, choose one of the following routes:

```bash
# Original HAM10000 image dataset (Kaggle API credentials required)
python main.py --mode download
python main.py --mode train --model hybrid

# Or use the included 28×28 pixel CSV
python run_full_pipeline.py
```

---

## 📖 Usage

### 1. Download Dataset

```bash
# Option A: Via CLI (requires kaggle.json in ~/.kaggle/)
python main.py --mode download

# Option B: Manual download
# Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# Extract into dataset/ folder
```

### 2. Train a Model

```bash
# Train the Hybrid model
python main.py --mode train --model hybrid

# Or train Custom CNN
python main.py --mode train --model custom_cnn

# Or train MobileNetV2
python main.py --mode train --model mobilenet_v2
```

### 3. Evaluate

```bash
python main.py --mode evaluate
```

### 4. Predict a Single Image

```bash
python main.py --mode predict --image path/to/lesion.jpg
```

This outputs:
- Predicted class with confidence
- Risk level (RED / YELLOW / GREEN)
- Clinical recommendations
- Grad-CAM visualisation saved to `results/`

### 5. Run the Web App

```bash
streamlit run app/streamlit_app.py
```

Before model artifacts exist, the Streamlit app provides project status,
setup instructions, and limitations without attempting inference. After a
model and label encoder are present, it provides:
- 📷 Image upload
- 📊 Real-time classification with probabilities
- 🔥 Grad-CAM heatmap overlay
- ⚠️ Clear model-confidence and medical-use limitations

The app includes an educational next-step guidance panel. It combines predicted
class with a 0.60 low-confidence cutoff; this cutoff is a prototype heuristic,
not a clinically calibrated threshold. Photo checks are simple heuristics and
may produce false warnings. These features are not clinically validated and do
not change the classifier's prediction.

### 6. Google Colab

Upload any notebook from `notebooks/` to [Google Colab](https://colab.research.google.com/) for GPU-accelerated training:

| Notebook | Architecture | Best For |
|----------|-------------|----------|
| `Skin_Cancer_CustomCNN_v3.ipynb` | Custom CNN | Learning / baseline |
| `Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb` | Hybrid | Architecture experiment |

---

## 🔧 Training Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌────────────┐
│  Load CSV   │───▶│ Label Encode │───▶│ Stratified Split │───▶│ Generators │
│ (metadata)  │    │  (7 classes) │    │  70/15/15 (%)    │    │ (augmented)│
└─────────────┘    └──────────────┘    └──────────────────┘    └─────┬──────┘
                                                                     │
                   ┌─────────────────────────────────────────────────┘
                   ▼
         ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐
         │  Stage 1: Train  │───▶│  Stage 2: Fine-  │───▶│  Evaluation  │
         │ (frozen backbone)│    │  tune top layers  │    │ (test set)   │
         │ 40 epochs, 1e-4  │    │ 10 epochs, 1e-5  │    │              │
         └──────────────────┘    └──────────────────┘    └──────┬───────┘
                                                                │
                   ┌────────────────────────────────────────────┘
                   ▼
         ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐
         │  Save Model &    │    │  Confusion Matrix │    │  Grad-CAM    │
         │  Label Encoder   │    │  & Metrics Report │    │  Heatmaps    │
         └──────────────────┘    └──────────────────┘    └──────────────┘
```

**Key Design Choices:**
- **Memory efficiency**: `flow_from_dataframe()` loads images in batches (no `np.zeros()` for entire dataset)
- **Class imbalance**: Automatic balanced class weights via `compute_class_weight`
- **Augmentation**: Rotation, zoom, shift, horizontal flip
- **Regularisation**: Dropout (0.2–0.5), Batch Normalisation, Label Smoothing (0.1)
- **Callbacks**: EarlyStopping (patience=5), ReduceLROnPlateau, ModelCheckpoint

---

## 📈 Results

### Local CSV pipeline run

The included 28×28 CSV pipeline completed locally in 11 epochs. Results below
come from its first held-out test run (1,503 images). The current training
workflow selects checkpoints by validation macro F1.

| Metric | Result |
|---|---:|
| Test accuracy | 56.02% |
| Weighted F1 | 60.53% |
| Macro F1 | 40.84% |
| Melanoma recall | 68.26% |

These results are for software demonstration only. The class-level errors and
dataset limitations make this model unsuitable for medical screening or
clinical decisions. The full classification report is saved in
`results/run_results.json` after training.

### Resume-ready project description

> Developed a skin-lesion classification prototype with a MobileNetV2
> transfer-learning option, Grad-CAM visualizations, confidence-aware next-step
> guidance, and image-quality prompts. Documented a reproducible evaluation
> workflow and research-only limitations.

---

## ⚙️ Configuration

All hyperparameters are centralised in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | (224, 224) | Input resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS_STAGE1` | 40 | Frozen backbone epochs |
| `EPOCHS_STAGE2` | 10 | Fine-tuning epochs |
| `LEARNING_RATE_STAGE1` | 1e-4 | Initial learning rate |
| `LEARNING_RATE_STAGE2` | 1e-5 | Fine-tuning learning rate |
| `EARLY_STOPPING_PATIENCE` | 5 | Epochs before early stop |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing factor |
| `TEST_SIZE` | 0.3 | Test+val fraction |

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow 2.x, Keras |
| Transfer Learning | MobileNetV2 (ImageNet) |
| Explainability | Grad-CAM |
| Data Science | NumPy, Pandas, scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Image Processing | OpenCV, Pillow |
| Web App | Streamlit |
| Dataset | HAM10000 (Kaggle) |

---

## ⚠️ Disclaimer

> **This is a research prototype for educational use only and is not for medical use.**
> It is **NOT** a substitute for professional medical diagnosis.
> Always consult a qualified dermatologist for clinical evaluation and treatment decisions.

---

## 📄 License

This project is for academic and research use. The HAM10000 dataset is provided under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

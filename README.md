# Partial Discharge Detection in Hydrogenerators via Ozone Analysis

This project was developed during my Scientific Initiation (PIBIC) at the Federal University of Pará (UFPA). It focuses on detecting partial discharges in hydroelectric generators by analyzing ozone concentration patterns.

## ⚡ The Project
Partial discharges are critical indicators of insulation failure in large generators. Since ozone is a byproduct of these discharges, we use simulated data of ozone concentration to identify equipment states.

## 🛠️ Tech Stack & Methodology
* **Data Processing:** Conversion of coordinates (x, y, color) into 2D images using linear interpolation.
* **Data Augmentation:** Robustness increased with Gaussian noise, blurring, and spatial shifts.
* **Feature Extraction:** Pre-trained **ResNet18** (CNN) used as a backbone to extract 512-dimensional feature vectors.
* **Classification:** **Support Vector Machine (SVM)** evaluated with 5-Fold Stratified Cross-Validation.

## 📂 Repository Structure
* `src/preprocess.py`: Interpolation and data augmentation.
* `src/extractor.py`: Feature extraction using PyTorch.
* `src/train_svm.py`: Training, cross-validation, and metrics.
* `requirements.txt`: List of necessary Python libraries.

## 🚀 How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Place raw CSV/ODS files in `data/raw/`.
3. Run: `python src/preprocess.py --input ./data/raw`
4. Run: `python src/extractor.py --input ./data/augmented`
5. Run: `python src/train_svm.py features.csv`

# Diabetic Retinopathy AI Diagnosis System (CAD)

A Computer-Aided Diagnosis (CAD) system utilizing Deep Learning and QWK-Weighted Ensemble Learning for automated detection and grading of Diabetic Retinopathy (DR) from retinal fundus images.

## Key Features
* Multi-Model Ensemble Architecture: Combines predictions from ResNet50, DenseNet201, EfficientNetV2L, and ConvNeXt_L.
* QWK-Weighted Voting: Dynamically weights each model's contribution based on its Quadratic Weighted Kappa (QWK) validation score.
* Test-Time Augmentation (TTA): Generates multiple augmented views per image during inference to maximize diagnostic reliability.
* Outlier Pre-Screening: Automatically flags known outlier images based on Exploratory Data Analysis (EDA) records.
* Interactive PyQt5 GUI: Displays class probability distributions, individual model votes, confidence metrics, and ground-truth comparisons.

## DR Severity Grading Scale
* 0: No DR
* 1: Mild NPDR
* 2: Moderate NPDR
* 3: Severe NPDR
* 4: Proliferative DR

## Project Structure
* GUI_Data_Augmentation.py: Main PyQt5 desktop application for real-time diagnosis.
* GUI_Data_Raspberry.py: Lightweight interface tailored for edge/Raspberry Pi deployment.
* Training_pytourch.py: Deep learning model architectures, image preprocessing, and training pipelines.
* EDA.py & EDA_Enhanced.py: Exploratory Data Analysis, PCA/UMAP/t-SNE feature visualization, and outlier detection scripts.
* run_evaluation.py: Comprehensive ensemble evaluation and error analysis generator.
* Work/: Contains model evaluation metrics (summary_metrics.csv), confusion matrices, and training histories.

## Setup and Installation

1. Clone the repository:
git clone https://github.com/RashedHEO/diabetic-retinopathy-cad-system.git
cd diabetic-retinopathy-cad-system

2. Install dependencies:
pip install -r requirements.txt

3. Download Pre-trained Weights:
Download the *_best.pth model checkpoints from:
https://drive.google.com/drive/folders/1mbwnyy3IGAm_PWUqCJ023aiZSPzzlEjP?usp=sharing
and place them inside the Work/ directory.

4. Run the Application:
python GUI_Data_Augmentation.py
# DermAssist
# DermAssist: Multi-Modal Skin Disease Classification using Dermoscopic and Clinical Images

## Overview

DermAssist is an AI-powered skin disease classification system designed to assist in the identification of various dermatological conditions using both dermoscopic and clinical images.

The project combines medical image processing, deep learning, feature engineering, lesion segmentation, and explainable AI techniques to create a robust diagnostic support system capable of classifying skin lesions into multiple disease categories.

The system leverages both close-range clinical photographs and dermoscopic images to improve diagnostic performance and emulate the multi-view assessment approach used by dermatologists.

---

## Problem Statement

Accurate skin disease diagnosis often requires:

* Visual examination from multiple distances
* Dermoscopic inspection
* Assessment of lesion asymmetry, border irregularity, color variation, and texture
* Expert interpretation

Manual diagnosis can be time-consuming and subjective. DermAssist aims to provide an AI-assisted diagnostic framework that learns from multiple image modalities and extracts clinically relevant information for disease classification.

---

## Objectives

* Develop an automated skin lesion classification system.
* Utilize dermoscopic and clinical images simultaneously.
* Perform lesion segmentation to isolate regions of interest.
* Extract medically relevant lesion characteristics.
* Incorporate Explainable AI (XAI) techniques.
* Classify lesions into multiple disease categories.
* Create a scalable framework suitable for future research and clinical applications.

---

## Datasets Used

### 1. MRA-MIDAS Dataset

Primary dataset used for disease classification.

Contains:

* Dermoscopic Images
* Clinical Images (approximately 6-inch and 1-foot views)
* Diagnostic Labels
* Multi-view lesion representation

Dataset Components:

* Dermoscopic image
* Clinical close-range image
* Clinical distant image
* Disease label hierarchy

---

### 2. ISIC 2018 Skin Lesion Segmentation Dataset

Used for lesion segmentation model training.

Contains:

* Dermoscopic images
* Expert-annotated lesion masks

Purpose:

* Train a U-Net segmentation network
* Generate lesion masks for images lacking ground-truth annotations
* Enable ROI extraction and feature engineering

---

## Disease Categories

The dataset contains lesions belonging to the following categories:

1. Malignant SCCIS
2. Malignant SCC
3. Malignant AK
4. Malignant BCC
5. Malignant Melanoma
6. Malignant Other
7. Benign Melanocytic Nevus
8. Benign Seborrheic Keratosis
9. Benign Dermatofibroma
10. Benign Hemangioma
11. Benign Fibrous Papule
12. Benign Other
13. Melanocytic Tumor
14. Other Melanocytic Lesion
15. Non-Neoplastic Inflammatory Lesion
16. Infectious Lesion
17. Additional Rare Lesion Categories

Total Classes: 17

---

## Methodology

### Phase 1: Data Preparation

* Dataset cleaning
* Label normalization
* Removal of incomplete samples
* Pairing dermoscopic and clinical images
* Creation of image triplets

Output:

* Structured multimodal dataset
* Training-ready image-label mappings

---

### Phase 2: Lesion Segmentation

Model:

* U-Net with ResNet34 Encoder

Training Dataset:

* ISIC 2018 Segmentation Dataset

Process:

1. Train segmentation model on ISIC masks.
2. Generate lesion masks for MRA-MIDAS images.
3. Extract lesion regions of interest (ROI).
4. Remove irrelevant background information.

Output:

* Binary lesion masks
* Cropped lesion ROIs

---

### Phase 3: Feature Extraction

Extracted medically relevant features inspired by the ABCD dermatology rule.

#### A – Asymmetry

Measures lesion symmetry across multiple axes.

#### B – Border Irregularity

Quantifies lesion boundary complexity.

#### C – Color Variation

Analyzes lesion color distribution using clustering techniques.

#### D – Diameter & Distribution

Captures lesion size-related characteristics.

Additional Features:

* Texture statistics
* Entropy
* Edge information
* Morphological descriptors

---

### Phase 4: Deep Feature Learning

Deep CNN backbones used:

* ResNet34
* EfficientNet
* Transfer Learning from ImageNet

Inputs:

* Dermoscopic images
* Clinical images

Output:

* High-level image embeddings

---

### Phase 5: Feature Fusion

Fusion combines:

* Dermoscopic image embeddings
* Clinical image embeddings
* Handcrafted lesion features

Advantages:

* Improved robustness
* Multi-view lesion understanding
* Better disease discrimination

---

### Phase 6: Classification

Multi-input neural network architecture:

Input:

* Dermoscopic image
* Clinical image
* Engineered lesion features

Output:

* Disease probability distribution across 17 classes

Loss Function:

* Cross Entropy Loss

Optimizer:

* Adam Optimizer

---

### Phase 7: Explainable AI (XAI)

To improve model interpretability:

#### Grad-CAM

Visualizes image regions influencing predictions.

#### Grad-CAM++

Improved localization of diagnostically relevant areas.

#### SHAP Analysis

Provides feature-level explanations.

Benefits:

* Transparency
* Trustworthiness
* Clinical interpretability

---

## System Architecture

Raw Images
↓
Preprocessing
↓
U-Net Segmentation
↓
Lesion ROI Extraction
↓
Feature Engineering
↓
CNN Feature Extraction
↓
Feature Fusion
↓
Disease Classification
↓
Explainability Module
↓
Prediction Output

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* PyTorch
* Torchvision
* Segmentation Models PyTorch

### Computer Vision

* OpenCV
* Scikit-Image

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Explainability

* Grad-CAM
* SHAP

---

## Key Contributions

* Multi-modal dermatology classification framework
* Integration of dermoscopic and clinical imaging
* Segmentation-based lesion isolation
* Medical feature extraction using ABCD principles
* Deep feature fusion architecture
* Explainable AI integration
* Research-oriented pipeline suitable for future extensions

---

## Future Work

* Fine-tuning segmentation models on MRA-MIDAS annotations
* Vision Transformer (ViT) based classification
* Clinical report generation using Large Language Models
* Mobile deployment for real-time diagnosis
* Integration with electronic health records

---

## Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

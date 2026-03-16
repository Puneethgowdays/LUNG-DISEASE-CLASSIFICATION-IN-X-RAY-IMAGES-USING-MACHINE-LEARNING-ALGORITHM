# 🫁 Lung Disease Classification in X-Ray Images

<p align="center">
<img src="[https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Python-3776AB%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)" />
<img src="[https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Flask-000000%3Fstyle%3Dfor-the-badge%26logo%3Dflask%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/OpenCV-5C3EE8%3Fstyle%3Dfor-the-badge%26logo%3Dopencv%26logoColor%3Dwhite)" />
</p>

## 📌 Project Overview

An automated diagnostic tool designed to classify chest X-ray images into four categories: **Lung Cancer, Tuberculosis, COVID-19, and Pneumonia**. This project utilizes a **ResNet50** CNN architecture integrated with **Radiomics** and **Genetic Algorithms** for high-precision medical imaging analysis.

## ⚙️ Core Pipeline

* 
**Data Acquisition:** Processes large-scale datasets including NIH Chest X-ray or Kaggle formats.


* 
**Preprocessing:** Includes RGB to Grayscale conversion, Median Filtering for noise removal, and Image Sharpening to enhance lung tissue visibility.


* 
**Feature Optimization:** Employs **Genetic Algorithms** to select the most discriminative radiomic features, reducing dimensionality while maintaining accuracy.


* 
**Classification:** Uses a deep **ResNet50** model with 50 layers and residual connections to identify infection markers.



## 🛠️ System Requirements

### Hardware

* 
**Processor:** Minimum 4 cores, 3.0 GHz.


* 
**RAM:** 8 GB (16 GB recommended).


* 
**GPU:** CUDA-compatible (NVIDIA GTX series).



### Software

* 
**OS:** Windows 10 / Linux / macOS.


* 
**Libraries:** `OpenCV`, `PyRadiomics`, `TensorFlow/Keras`, `Flask`, `Scikit-Learn`.



## 🧪 Testing & Validation

The system was verified through rigorous testing modules:
| Test Type | Objective | Status |
| :--- | :--- | :--- |
| **Unit Testing** | Individual component isolation (Preprocessing, Feature Extraction) | **PASS**  |
| **System Testing** | End-to-end diagnosis and UI performance | **PASS**  |
| **Integration Testing** | GA and Classification module interactions | **PASS**  |

## 👥 Contributors

* 
**Puneeth Gowda Y S** 

* 
**Guide:** Mr. Sridhara N, Assistant Professor, Dept. of IS&E, BGSIT 

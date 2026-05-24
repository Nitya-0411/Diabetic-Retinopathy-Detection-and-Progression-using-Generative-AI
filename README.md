# Generative AI for Diabetic Retinopathy Detection and Progression

An AI-powered system for **Diabetic Retinopathy (DR) classification and disease progression simulation** using **EfficientNet-B0** and **CycleGAN**.

This project predicts the severity stage of Diabetic Retinopathy from retinal fundus images and simulates how the disease may progress over time using Generative AI techniques.

---

## Project Overview

Diabetic Retinopathy is a major cause of vision loss among diabetic patients. Early diagnosis and monitoring are important to prevent blindness.

This project combines:

- **EfficientNet-B0** → DR severity classification
- **CycleGAN** → Disease progression simulation
- **Deep Learning + Generative AI** → Prediction and visualization

The system classifies retinal images into DR stages and generates future progression images to estimate disease advancement.

---

## Features

✔ Diabetic Retinopathy classification  
✔ Severity prediction (5 stages)  
✔ Disease progression simulation  
✔ Synthetic retinal image generation using CycleGAN  
✔ Image preprocessing & augmentation  
✔ Deep learning-based diagnosis support  

---

## DR Severity Classes

The model predicts:

```text
0 → No DR
1 → Mild
2 → Moderate
3 → Severe
4 → Proliferative DR
```

---

## Dataset

Dataset contains approximately:

- 250 retinal fundus images
- Collected from external hospital sources
- Organized into multiple DR stages

Dataset preprocessing includes:

- Image resizing (224×224)
- Normalization
- Data augmentation
- Rotation
- Brightness adjustment
- Flipping

---

## Technologies Used

Python  
PyTorch  
Google Colab  
EfficientNet-B0  
CycleGAN  
Matplotlib  
NumPy  
OpenCV  

---

## Models Used

### 1. EfficientNet-B0

Used for:

- Feature extraction
- Retinal image classification
- DR severity prediction

Saved trained model:

```text
efficientnetb0_dr_final.pth
```

---

### 2. CycleGAN

Used for:

- Mild → Moderate progression
- Moderate → Severe progression
- Synthetic disease progression generation

---

## Project Workflow

```text
Input Retinal Image
        ↓
Image Preprocessing
        ↓
EfficientNet-B0 Classification
        ↓
DR Severity Prediction
        ↓
CycleGAN Progression Simulation
        ↓
Future Disease Visualization
```

---

## Folder Structure

```text
PROJECT/
│
├── dataset/
│   ├── image_dataset.py
│   └── __init__.py
│
├── GenAI/
│   ├── Mild_Moderate/
│   └── Moderate_Severe/
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── __init__.py
│
├── runs/
│   ├── Mild_to_Moderate/
│   └── Moderate_to_Severe/
│
├── utils/
│   ├── buffer.py
│   ├── lambda_lr.py
│   └── weight_init.py
│
├── efficientnetb0_dr_final.pth
├── main.py
├── requirements.txt
└── New_interface_UI.ipynb
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/repository-name.git
```

Move into project folder:

```bash
cd repository-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

Run:

```bash
python main.py
```

or open:

```text
New_interface_UI.ipynb
```

using Jupyter Notebook / Google Colab.

---

## Results

Classification achieved high accuracy for DR stage prediction using EfficientNet-B0.

Evaluation performed using:

- Accuracy curves
- Loss curves
- Confusion matrix
- Classification report

CycleGAN successfully generated realistic progression images between disease stages.

---

## Future Scope

Possible improvements:

- Real-time hospital integration
- Explainable AI for diagnosis
- Personalized progression forecasting
- Collaboration with ophthalmologists
- Deployment as web application

---

## Contributors

- K Anandi Raghavi
- T Maha Lakshmi
- V Nitya Vaishnavi

Guide:

Mrs. Y Sravani Devi

---

## Note

Developed as an academic project.

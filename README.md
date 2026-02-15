# Sonar Object Classification (Rock vs Bomb)

## Project Overview
This project uses **sonar signal data** to classify an object as either:

- **Rock**
- **Bomb (Mine)**

The goal is to build a **simple machine learning pipeline** that takes sonar readings as input and predicts the object type.

This is a **binary classification** problem.

---

## Dataset
- The dataset contains **sonar signal strengths**
- Each sample has **multiple numeric features**
- The label is:
  - `R` → Rock  
  - `M` → Bomb (Mine)

Example format:
feature_1, feature_2, ..., feature_n, label


---

## Tech Stack
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib / Seaborn (optional)

---

## Project Structure
sonar-ml-project/
│
├── data/
│ └── sonar.csv
│
├── notebooks/
│ └── exploration.ipynb
│
├── src/
│ ├── load_data.py
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│
├── models/
│ └── model.pkl
│
├── README.md
└── requirements.txt


---

## Machine Learning Workflow

### 1. Data Loading
- Load the sonar dataset
- Separate features (X) and labels (y)

**File:** `load_data.py`

---

### 2. Data Preprocessing
- Encode labels (`R` → 0, `M` → 1)
- Scale numeric features
- Split data into training and test sets

**File:** `preprocess.py`

---

### 3. Model Training
- Train a classifier (for example):
  - Logistic Regression
  - Support Vector Machine
  - Random Forest
- Fit model on training data

**File:** `train.py`

---

### 4. Model Evaluation
- Evaluate the model using:
  - Accuracy
  - Confusion matrix
  - Precision and recall
- Print results clearly

**File:** `evaluate.py`

---

### 5. Model Saving (Optional)
- Save trained model using `pickle` or `joblib`
- Load model later for predictions

---

## Workflow Diagram
Raw Sonar Data
↓
Preprocessing & Scaling
↓
Train / Test Split
↓
Model Training
↓
Model Evaluation
↓
Prediction


---

## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
Train the model

python src/train.py
Evaluate the model

python src/evaluate.py
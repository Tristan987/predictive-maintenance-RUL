# Predictive Maintenance Using Remaining Useful Life (RUL) Regression

This repository contains the final version of our Machine Learning project for **ESILV – A4 Engineering (MMN)**.  
The goal is to build a predictive maintenance model capable of estimating the **Remaining Useful Life (RUL)** of turbofan engines using multivariate time-series sensor data.
Team contribution : Tristan 80 / Bruna 10 / Godeffroy 10

---

## 📌 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Dataset](#2-dataset)
- [3. Methodology](#3-methodology)
- [4. Models & Results](#4-models--results)
- [5. Repository Structure](#5-repository-structure)
- [6. How to Run](#6-how-to-run)
- [7. Authors](#7-authors)

---

## 1. Project Overview

Aircraft engines operate under extreme conditions and require continuous monitoring.  
Unexpected failures lead to:

- Safety risks  
- Costly unplanned maintenance  
- Delays and operational disruptions  

To address this issue, we developed a **Remaining Useful Life (RUL) regression model**, capable of predicting how many cycles remain before engine failure.  
This enables a transition from **scheduled maintenance** to **condition-based maintenance**, improving safety and reducing operational costs.

This project includes:

- Full dataset exploration  
- RUL target construction  
- Feature selection and correlation analysis  
- Baseline and ensemble model training  
- Hyperparameter tuning  
- Final evaluation on the official test set  

---

## 2. Dataset

We use three dataset files:

```
PM_train.csv  
PM_test.csv  
PM_truth.csv
```

Each row corresponds to an engine cycle and includes:

- Engine ID  
- Cycle index  
- 3 operational settings  
- 21 sensor measurements  
- Target: Remaining Useful Life (RUL) — computed for the training set  

The test set is truncated and requires the use of `PM_truth.csv` to evaluate model performance.

---

## 3. Methodology

Our process follows a structured ML workflow:

### **1. Data loading & inspection**
- Detect dataset shape  
- Check missing values  
- Analyze sensor behavior  
- Visualize degradation trends  

### **2. RUL target engineering**
For every engine in the training set:

```
RUL = last_cycle_of_engine – current_cycle
```

### **3. Feature selection & correlation**
- Remove constant or irrelevant sensors  
- Study multicollinearity using a correlation heatmap  
- Normalize features when necessary  

### **4. Train/Validation split**
Split is performed **by engine ID** to avoid data leakage.

### **5. Models tested**
We evaluated 4 regression models:

- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Extra Trees Regressor  

### **6. Hyperparameter tuning**
Performed using `GridSearchCV` to optimize ensemble models.

### **7. Final training**
Best model retrained on the entire training set.

---

## 4. Models & Results

Performance on the **test set**:

| Model                     | RMSE  | MAE   | R²     |
|---------------------------|------:|------:|-------:|
| Linear Regression         | 44.34 | 34.05 | 0.5696 |
| Random Forest             | 41.40 | 29.62 | 0.6249 |
| Gradient Boosting         | 41.30 | 29.73 | 0.6266 |
| Extra Trees               | 41.13 | 29.45 | 0.6297 |

### 🔥 **Best Model: Extra Trees Regressor**  
It achieved the lowest RMSE and highest R² value, making it the most accurate model for RUL prediction in our experiments.

---

## 5. Repository Structure

```
predictive-maintenance-RUL/
│
├── ML Project - final version.ipynb
├── PM_train.csv
├── PM_test.csv
├── PM_truth.csv
└── README.md
```

*(This structure may evolve if additional folders such as `/figures` or `/src` are added.)*

---

## 6. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Launch the notebook
```bash
jupyter notebook "ML Project - final version.ipynb"
```

The notebook includes:

1. Exploratory Data Analysis  
2. RUL computation  
3. Feature selection  
4. Baseline models  
5. Ensemble models  
6. Hyperparameter tuning  
7. Final evaluation  

---

## 7. Authors

**Tristan Gaveau**  
**Godeffroy Gonnin**  
**Bruna Fernandes Parada**  
*ESILV – A4 Engineering (MMN)*

---

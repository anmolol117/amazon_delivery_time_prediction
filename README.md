# Amazon Delivery Time Prediction — Version 03 

## Overview

This project focuses on predicting **Amazon order delivery time (ETA)** using advanced **machine learning and deep learning techniques** on a rich dataset containing **spatial, temporal, operational, and contextual delivery features**.

The primary objective is to improve **last-mile logistics efficiency**, optimize delivery operations, and enhance customer satisfaction through accurate **Estimated Time of Arrival (ETA)** predictions.

**Version 03** significantly improves upon the previous baseline model by introducing:

- Advanced feature engineering  
- Geographically accurate distance calculations  
- Improved data cleaning  
- Enhanced neural network architecture  
- Ensemble learning via model stacking  

This resulted in a **44.6% improvement in RMSE** over the baseline model.

---

## Dataset Features

The model utilizes a wide variety of delivery-related features.

### Spatial Features
- Store latitude and longitude
- Drop-off latitude and longitude
- **Haversine Distance** (geographically accurate Earth-surface distance)
- **Manhattan Distance** (urban routing approximation)

### Temporal Features
- Order time
- Pickup time
- Hour of day
- Rush-hour flag
- Late-night delivery flag
- Weekend indicator

### Contextual & Operational Features
- Weather conditions
- Traffic density levels
- Vehicle type
- Delivery area
- Product category
- Delivery agent age
- Delivery agent rating

---

## Key Improvements in Version 03

### 1. Advanced Distance Engineering
Traditional Euclidean distance on latitude/longitude was replaced with:

- **Haversine Distance** for geographically correct Earth-surface travel estimation
- **Manhattan Distance** to better approximate real-world urban road routing

These changes significantly improved spatial understanding for delivery prediction.

---

### 2. Improved Data Cleaning
Invalid coordinate values such as **(0,0)** were handled using **area-based imputation**, reducing spatial noise and preventing corrupted geolocation data from negatively affecting predictions.

---

### 3. Rich Temporal Feature Engineering
New time-aware features were introduced:

- Hour of the day
- Rush-hour indicator
- Late-night delivery flag
- Weekend flag

These features help the model better capture demand fluctuations and traffic patterns.

---

### 4. Expanded Feature Utilization
Previously excluded but informative features were reintroduced:

- **Agent Age**
- **Agent Rating**
- **Product Category**

This improved the model’s ability to capture delivery efficiency and order complexity.

---

### 5. Optimized Feature Selection
Polynomial feature expansion was removed to reduce dimensionality bloat and overfitting.

Since tree-based models naturally learn nonlinear interactions, this improved generalization performance.

---

### 6. Improved Deep Learning Architecture
The neural network was enhanced using:

- **Wider architecture (512 neurons)**
- **Swish activation function**
- **Reduced dropout**

These improvements increased learning capacity and optimization stability.

---

### 7. Stacking Ensemble Learning
A **4-model stacking ensemble** was implemented using:

- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Improved Deep Neural Network (DNN)**
- **RidgeCV Meta-Learner**

The ensemble combines strengths from multiple models to improve prediction robustness and reduce overall error.

---

## Models Implemented

The following models were trained and evaluated:

- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Improved Deep Neural Network (TensorFlow/Keras)**
- **Stacking Ensemble Model (Best Performing Model)**

### Evaluation Metrics
Performance was evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

---

## Model Performance

| Model | RMSE | MAE | R² |
|--------|------|------|------|
| V2 Baseline (DNN + XGB Stack) | 40.9900 | — | — |
| V3 Random Forest | 23.5539 | 18.0241 | 0.7957 |
| V3 Gradient Boosting | 22.8381 | 17.7141 | 0.8079 |
| V3 Improved DNN | 23.8696 | 18.2578 | 0.7902 |
| **V3 Stacking Ensemble** | **22.7151** | **17.5821** | **0.8100** |

---

## Best Performing Model

### **V3 Stacking Ensemble**

| Metric | Value |
|---------|-------|
| **RMSE** | **22.7151** |
| **MAE** | **17.5821** |
| **R² Score** | **0.8100** |
| **Improvement over V2** | **44.6%** |

---

## Tech Stack

- **Python**
- **Scikit-learn**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

---

## Real-World Applications

This predictive system can be used for:

- **Estimated Time of Arrival (ETA) Prediction**
- **Last-Mile Delivery Optimization**
- **Logistics & Route Planning**
- **Demand-Aware Delivery Scheduling**
- **Customer Experience Enhancement**

---

## Results & Conclusion

Version 03 demonstrates how **domain-aware feature engineering**, **improved preprocessing**, and **ensemble learning** can significantly enhance delivery time prediction performance.

By combining:

- Spatial intelligence  
- Temporal behavior modeling  
- Improved neural networks  
- Ensemble learning techniques  

the system achieved a **44.6% reduction in RMSE**, improving from **40.99 → 22.71**, while achieving an **R² score of 0.81**.

This project highlights the importance of combining **data preprocessing, feature engineering, and model ensembling** to build highly accurate real-world predictive systems.

# 🧠 Stroke Prediction Using Machine Learning

A Machine Learning based web application that predicts the likelihood of a stroke based on patient health and lifestyle data.  
This project uses a **Random Forest Classifier** and provides predictions through a **simple user interface**.

---

## 🚀 Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide.  
Early prediction can help in taking preventive measures and improving patient outcomes.

This project aims to:
- Analyze medical and lifestyle features
- Predict the probability of stroke
- Provide an easy-to-use prediction interface

---

## 🛠️ Tech Stack Used

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Model:** Random Forest Classifier  
- **Data Processing:** Pandas, NumPy  
- **Model Storage:** Pickle  
- **Frontend / UI:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure
stroke-prediction/
│
├── app.py # Streamlit web application
├── train_model.py # Model training script
├── stroke_data.csv # Dataset
├── model.pkl # Trained ML model
├── encoders.pkl # Label encoders
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 📊 Dataset Description

The dataset contains medical and lifestyle-related features such as:

- Gender  
- Age  
- Hypertension  
- Heart disease  
- Ever married  
- Work type  
- Residence type  
- Average glucose level  
- BMI  
- Smoking status  

**Target Variable:**  
- `stroke` (0 = No Stroke, 1 = Stroke)

---

## ⚙️ How the Model Works

1. Load and clean the dataset  
2. Handle missing values  
3. Encode categorical variables using Label Encoding  
4. Split data into training and testing sets  
5. Train a Random Forest Classifier  
6. Evaluate accuracy  
7. Save the trained model and encoders  



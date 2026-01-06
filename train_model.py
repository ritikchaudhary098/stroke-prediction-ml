# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
data = pd.read_csv("stroke_data.csv")

# Remove missing values
data = data.dropna()

# ---------------- ENCODE CATEGORICAL DATA ----------------
categorical_cols = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le   # save encoder for later use

# ---------------- SPLIT FEATURES & TARGET ----------------
X = data.drop(['id', 'stroke'], axis=1)
y = data['stroke']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Training Complete")
print("🎯 Accuracy:", accuracy)

# ---------------- SAVE MODEL & ENCODERS ----------------
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("💾 model.pkl and encoders.pkl saved successfully")


import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
file_path = "data/Project_datasheet_2019-2020.xlsx"
df = pd.read_excel(file_path, sheet_name="Project datasheet")

# 2. Select useful columns
columns = [
    "CropType",
    "CropDays",
    "Soil Moisture",
    "Soil Temperature",
    "Temperature",
    "Humidity"
]

X = df[columns]
y = df["Irrigation(Y/N)"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build and train the neural network
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 7. Save the trained model and scaler
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/irrigation_dnn.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(columns, "models/feature_columns.pkl")

print("\nModel training completed!")
print("Model saved in the models folder.")

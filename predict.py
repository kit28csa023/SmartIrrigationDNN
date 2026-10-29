
import pandas as pd
import joblib

# Load the trained model and supporting files
model = joblib.load("models/irrigation_dnn.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/feature_columns.pkl")

print("\n=== SMART IRRIGATION PREDICTION ===")

crop = int(input("Crop type (1=Paddy, 2=Ground Nuts): "))
days = float(input("Crop age in days: "))
moisture = float(input("Soil moisture reading: "))
soil_temp = float(input("Soil temperature: "))
temp = float(input("Air temperature: "))
humidity = float(input("Humidity percentage: "))

# Prepare input in the same format as training data
data = pd.DataFrame(
    [[crop, days, moisture, soil_temp, temp, humidity]],
    columns=columns
)

# Scale and predict
data_scaled = scaler.transform(data)
prediction = int(model.predict(data_scaled)[0])

print("\n--- RESULT ---")

if prediction == 1:
    print("Irrigation is recommended.")
else:
    print("Irrigation is not recommended.")

print("Prediction completed!")

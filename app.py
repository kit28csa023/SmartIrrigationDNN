
from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained DNN model
model = joblib.load("models/irrigation_dnn.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/feature_columns.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Irrigation System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #dff5e1, #e5f0ff);
            margin: 0;
            padding: 30px 15px;
        }
        .container {
            max-width: 650px;
            margin: auto;
            background: white;
            padding: 28px;
            border-radius: 18px;
            box-shadow: 0 8px 25px #0002;
        }
        h1 { color: #176b45; text-align: center; }
        .subtitle { text-align: center; color: #555; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        label { font-weight: bold; display: block; margin-bottom: 7px; }
        input, select {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            border: 1px solid #bbb;
            border-radius: 8px;
            font-size: 15px;
        }
        button {
            width: 100%;
            margin-top: 22px;
            padding: 14px;
            background: #16834a;
            color: white;
            border: none;
            border-radius: 9px;
            font-size: 17px;
            cursor: pointer;
        }
        button:hover { background: #106738; }
        .result {
            margin-top: 22px;
            padding: 16px;
            background: #e7f7eb;
            color: #155b31;
            border-radius: 10px;
            text-align: center;
        }
        .error { background: #ffeded; color: #a32222; }
        .note { color: #666; font-size: 12px; margin-top: 20px; }
        @media (max-width: 480px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🌱 Smart Irrigation</h1>
    <p class="subtitle">IoT-Based Irrigation Prediction using DNN</p>

    <form method="POST">
        <div class="grid">
            <div>
                <label>Crop Type</label>
                <select name="crop" required>
                    <option value="1">Paddy</option>
                    <option value="2">Ground Nuts</option>
                </select>
            </div>
            <div>
                <label>Crop Age (days)</label>
                <input type="number" name="days" min="1" required>
            </div>
            <div>
                <label>Soil Moisture</label>
                <input type="number" name="moisture" step="any" required>
            </div>
            <div>
                <label>Soil Temperature (°C)</label>
                <input type="number" name="soil_temp" step="any" required>
            </div>
            <div>
                <label>Air Temperature (°C)</label>
                <input type="number" name="temp" step="any" required>
            </div>
            <div>
                <label>Humidity (%)</label>
                <input type="number" name="humidity" min="0"
                       max="100" step="any" required>
            </div>
        </div>
        <button type="submit">Predict Irrigation</button>
    </form>

    {% if result %}
    <div class="result {% if error %}error{% endif %}">
        <h3>{{ result }}</h3>
    </div>
    {% endif %}

    <p class="note">
        This is a machine-learning prediction based on the training dataset.
        It is not a substitute for checking actual field conditions.
        No water pump is controlled by this webpage.
    </p>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    error = False

    if request.method == "POST":
        try:
            values = [
                int(request.form["crop"]),
                float(request.form["days"]),
                float(request.form["moisture"]),
                float(request.form["soil_temp"]),
                float(request.form["temp"]),
                float(request.form["humidity"])
            ]

            if values[0] not in (1, 2):
                raise ValueError("Choose a valid crop type.")
            if values[1] < 1:
                raise ValueError("Crop age must be at least 1 day.")
            if not 0 <= values[5] <= 100:
                raise ValueError("Humidity must be between 0 and 100.")

            data = pd.DataFrame([values], columns=columns)
            scaled_data = scaler.transform(data)
            prediction = int(model.predict(scaled_data)[0])

            if prediction == 1:
                result = "💧 Irrigation is recommended."
            else:
                result = "✅ Irrigation is not recommended."

        except ValueError as e:
            result = str(e) or "Please enter valid values."
            error = True
        except Exception:
            result = "Unable to make a prediction. Please check your inputs."
            error = True

    return render_template_string(HTML, result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)

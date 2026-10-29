# IoT-Based Smart Irrigation Prediction using DNN

## Project Overview
This project predicts whether irrigation is recommended based on crop and environmental data using a Deep Neural Network (DNN) classification approach.

## Objectives
- Analyze soil and environmental parameters.
- Predict irrigation requirements using machine learning.
- Evaluate model performance.
- Provide a simple web interface for irrigation prediction.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib
- HTML and CSS

## Input Features
- Crop Type
- Crop Age
- Soil Moisture
- Soil Temperature
- Air Temperature
- Humidity

## Output
The model predicts one of two classes:
- Irrigation recommended
- Irrigation not recommended

## Model Performance
The current test accuracy was **96.67%** on 30 test records. This is a preliminary result from a small dataset and does not establish real-world reliability.

## How to Run
1. Install Python 3.
2. Install the required libraries:
   `python -m pip install -r requirements.txt`
3. Train the model:
   `python train_model.py`
4. Start the web application:
   `python app.py`
5. Open `http://127.0.0.1:5000` in your browser.

## Current Scope
This is a software prototype using an existing Excel dataset. Live sensor integration and automatic pump control are planned future improvements and are not currently implemented.

## Repository
https://github.com/kit28csa023/SmartIrrigationDNN

import os
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# -----------------------------
# Flask initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Base directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model-2.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "standard_scaler.joblib"))

# -----------------------------
# Feature list (must match training)
# -----------------------------
FEATURE_COLUMNS = [
    "PackVoltage_V",
    "CellVoltage_V",
    "DemandVoltage_V",
    "ChargeCurrent_A",
    "DemandCurrent_A",
    "SOC_%",
    "MaxTemp_C",
    "MinTemp_C",
    "AvgTemp_C",
    "AmbientTemp_C",
    "InternalResistance_mOhm",
    "StateOfHealth_%",
    "VibrationLevel_mg",
    "MoistureDetected",
    "ChargePower_kW",
    "Pressure_kPa",
    "ChargingStage_Handshake",
    "ChargingStage_Parameter_Config",
    "ChargingStage_Recharge",
    "BMS_Status_OK",
    "BMS_Status_Warning"
]

# -----------------------------
# Home route
# -----------------------------
@app.route("/")
def home():
    return "EV Battery Thermal Runaway Prediction API is running."

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        df = pd.DataFrame([data])

        # Ensure correct feature alignment
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # Scale input
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_class_0": float(probability[0][0]),
            "probability_class_1": float(probability[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Render PORT binding
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("Starting server on port:", port)
    app.run(host="0.0.0.0", port=port)

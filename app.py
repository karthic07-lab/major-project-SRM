import os
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import joblib

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
# Home
# -----------------------------
@app.route("/")
def home():
    return "EV Battery Thermal Runaway Prediction API is running."

# -----------------------------
# API prediction (JSON)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    prob = model.predict_proba(df_scaled)

    return jsonify({
        "prediction": int(pred[0]),
        "probability_class_0": float(prob[0][0]),
        "probability_class_1": float(prob[0][1])
    })

# -----------------------------
# Browser UI
# -----------------------------
@app.route("/ui", methods=["GET", "POST"])
def ui():
    result = None

    if request.method == "POST":
        form_data = {}
        for col in FEATURE_COLUMNS:
            form_data[col] = float(request.form.get(col, 0))

        df = pd.DataFrame([form_data])
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)
        prob = model.predict_proba(df_scaled)

        result = {
            "prediction": int(pred[0]),
            "safe_prob": round(prob[0][0], 3),
            "risk_prob": round(prob[0][1], 3)
        }

    return render_template_string("""
    <html>
    <head>
        <title>EV Battery Thermal Runaway Predictor</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            input { width: 200px; margin-bottom: 8px; }
            button { padding: 10px; }
            .result { margin-top: 20px; font-size: 18px; }
        </style>
    </head>
    <body>
        <h2>EV Battery Thermal Runaway Prediction</h2>
        <form method="post">
            {% for col in features %}
                <label>{{ col }}</label><br>
                <input type="number" step="any" name="{{ col }}" required><br>
            {% endfor %}
            <br>
            <button type="submit">Predict</button>
        </form>

        {% if result %}
        <div class="result">
            <h3>Result</h3>
            <p><b>Prediction:</b> {{ result.prediction }}</p>
            <p><b>Safe Probability:</b> {{ result.safe_prob }}</p>
            <p><b>Thermal Runaway Risk Probability:</b> {{ result.risk_prob }}</p>
        </div>
        {% endif %}
    </body>
    </html>
    """, features=FEATURE_COLUMNS, result=result)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

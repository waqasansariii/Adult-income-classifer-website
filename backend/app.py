from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = joblib.load("saved_models/best_xgboost_model_for_adult.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        input_df = pd.DataFrame([data])

        # Convert to numeric
        numeric_columns = ["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
        input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Add derived features
        input_df["capital.gain_log"] = np.log1p(input_df["capital.gain"])
        input_df["capital.loss_log"] = np.log1p(input_df["capital.loss"])

        print("Prepared DataFrame:", input_df)

        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    BACKEND_PORT = 8000
    app.run(debug=True, host="0.0.0.0", port=BACKEND_PORT)

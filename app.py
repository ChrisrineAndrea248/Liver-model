from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("liver_model.pkl")
scaler = joblib.load("scaler.pkl")

# In-memory prediction summary (will reset on restart)
prediction_summary = {
    'Liver Disease Detected': 0,
    'No Liver Disease': 0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_summary  # allow modification of global report

    try:
        # List of features (must match model input order)
        fields = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins',
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]

        # Collect input from form
        input_data = [float(request.form.get(f)) for f in fields]
        input_dict = dict(zip(fields, input_data))

        # Predict using trained model
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction] * 100

        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"

        # Update report data
        prediction_summary[result] += 1

        # Return prediction and summary to template
        return render_template('index.html',
                               prediction=result,
                               probability=round(probability, 2),
                               input_data=input_dict,
                               summary=prediction_summary)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

# Required for Render deployment (host must be 0.0.0.0 and port from env)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

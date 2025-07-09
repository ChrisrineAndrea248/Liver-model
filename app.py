from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("liver_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form inputs in the correct order
        input_data = [
            float(request.form.get('Age')),
            float(request.form.get('Gender')),
            float(request.form.get('Total_Bilirubin')),
            float(request.form.get('Direct_Bilirubin')),
            float(request.form.get('Alkaline_Phosphotase')),
            float(request.form.get('Alamine_Aminotransferase')),
            float(request.form.get('Aspartate_Aminotransferase')),
            float(request.form.get('Total_Proteins')),
            float(request.form.get('Albumin')),
            float(request.form.get('Albumin_and_Globulin_Ratio'))
        ]

        # Scale and predict
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        # Format result
        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

# -------------------------
# For deployment on Render
# -------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

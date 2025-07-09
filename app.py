from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("liver_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define feature fields in order
        fields = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins',
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]

        # Extract input values
        input_data = [float(request.form.get(f)) for f in fields]
        input_dict = dict(zip(fields, input_data))

        # Preprocess and predict
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        # Format result
        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"

        # Render result on same page
        return render_template('index.html', prediction=result, input_data=input_dict)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

# For local + Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

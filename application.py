from flask import Flask, render_template, request
import pickle
import numpy as np
import os

application = Flask(__name__)
app = application

# get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model and scaler (RELATIVE PATH)
scaler = pickle.load(open(os.path.join(BASE_DIR, "Model", "standardScaler.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "Model", "modelForPrediction.pkl"), "rb"))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[
            Pregnancies, Glucose, BloodPressure,
            SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age
        ]])

        prediction = model.predict(new_data)

        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        return render_template('single_prediction.html', result=result)

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

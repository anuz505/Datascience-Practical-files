from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

# Set up the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = joblib.load(os.path.join(BASE_DIR, 'models', 'grade_predictor_model.joblib'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()
    # Convert the data into a DataFrame
    input_data = pd.DataFrame([data])
    # Make a prediction
    prediction = model.predict(input_data)
    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

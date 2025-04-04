import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

# Load the trained model
with open('homeprices.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    area = float(request.form['area'])
    
    # Predict price
    prediction = model.predict(np.array([[area]]))[0]
    
    # Return result
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)

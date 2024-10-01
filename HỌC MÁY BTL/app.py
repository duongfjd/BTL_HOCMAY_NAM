import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the models
models = {
    'linear': joblib.load('HỌC MÁY BTL/linear_regression_model.pkl'),
    'ridge': joblib.load('HỌC MÁY BTL/ridge_regression_model.pkl'),
    'nn': joblib.load('HỌC MÁY BTL/neural_network_model.pkl')
}

# Route to load the HTML template
@app.route('/')
def demo():
    return render_template('demo.html')

# Route for prediction with dynamic algorithm
@app.route('/predict/<algorithm>', methods=['POST'])
def predict(algorithm):
    # Get the corresponding model
    model = models.get(algorithm)
    if not model:
        return jsonify({'error': 'Algorithm not supported'}), 400

    # Get data from the request
    data = request.get_json()

    try:
        # Extract input features
        performance = float(data['performance'])
        storage_capacity = float(data['storage_capacity'])
        camera_quality = float(data['camera_quality'])
        battery_life = float(data['battery_life'])
        weight = float(data['weight'])
        age = int(data['age'])
    except ValueError:
        return jsonify({'error': 'Invalid input values'}), 400

    # Create a feature array for prediction
    features = np.array([[performance, storage_capacity, camera_quality, battery_life, weight, age]])

    try:
        # Make the prediction
        prediction = model.predict(features)

        # Return the result as a JSON response
        return jsonify({
            'result': prediction.tolist()[0],  # Trả về kết quả
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)

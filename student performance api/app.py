from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('student_performance_model.pkl')

# API endpoint
@app.route('/predict', methods=['POST'])
def predict_performance():
    data = request.get_json()

    # Receive input values
    avg_previous_score = data.get('AvgPreviousScore')
    num_quizzes_taken = data.get('NumQuizzesTaken')
    last_score = data.get('LastScore')

    # Set a static value for AvgTimeSpent
    avg_time_spent = 30.0

    # Check for missing values
    if None in (avg_previous_score, num_quizzes_taken, last_score):
        return jsonify({'error': 'Missing input data'}), 400

    # Prepare data for the model
    features = np.array([[avg_previous_score, avg_time_spent, num_quizzes_taken, last_score]])

    # Predict the performance
    prediction = model.predict(features)

    return jsonify({
        'PredictedPerformance': round(prediction[0], 2)
    })

if __name__ == '__main__':
    app.run(debug=True)

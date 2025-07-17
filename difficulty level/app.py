# app.py

from flask import Flask, request, jsonify
import joblib

# Load trained model and label encoder
model = joblib.load("question_difficulty_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

@app.route("/predict_difficulty", methods=["POST"])
def predict_difficulty():
    data = request.get_json()
    
    question = data.get("question", "")
    optionA = data.get("optionA", "")
    optionB = data.get("optionB", "")
    optionC = data.get("optionC", "")
    optionD = data.get("optionD", "")
    
    full_text = f"{question} {optionA} {optionB} {optionC} {optionD}"
    prediction = model.predict([full_text])[0]
    difficulty = label_encoder.inverse_transform([prediction])[0]
    
    return jsonify({"difficulty": difficulty})

if __name__ == '__main__':
    app.run(debug=True, port=5001)


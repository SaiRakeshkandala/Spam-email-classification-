from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
        transformed_data = vectorizer.transform([data]).toarray()
            prediction = model.predict(transformed_data)
                return jsonify({'prediction': 'spam' if prediction[0] == 1 else 'ham'})

                if __name__ == '__main__':
                    app.run(debug=True)
                    
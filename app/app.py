from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/spam_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/', methods=["GET", "POST"])
def index():
    prediction = None
    message = ""
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        result = model.predict(data)
        prediction = "🚨 This message is Spam!" if result[0] == 1 else "✅ This message is Not Spam."
    return render_template("index.html", prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)

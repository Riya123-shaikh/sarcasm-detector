from flask import Flask, render_template, request
import joblib
from preprocess import clean_tweet

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sarcasm_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        tweet = request.form["tweet"]
        cleaned = clean_tweet(tweet)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        result = "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

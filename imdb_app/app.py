from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        review = request.form["review"]
        review_vec = vectorizer.transform([review])
        pred = model.predict(review_vec)[0]
        
        prediction = "Positive 😊" if pred == 1 else "Negative 😞"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
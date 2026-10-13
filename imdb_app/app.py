
from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle
import re
import os
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# -------------------------------
# Paths and Model Loading
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_LR.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_reviews.csv")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]


# -------------------------------
# Download Sample Dataset
# -------------------------------
@app.route("/download-sample")
def download_sample():
    if not os.path.isfile(SAMPLE_PATH):
        return "Sample dataset not found.", 404

    return send_file(
        SAMPLE_PATH,
        as_attachment=True,
        download_name="sample_reviews.csv",
        mimetype="text/csv"
    )


# -------------------------------
# Clean Text
# -------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text.lower()


# -------------------------------
# Get Top Contributing Words
# -------------------------------
def get_top_words(vector):
    vector = vector.toarray()[0]

    contributions = vector * weights
    sorted_idx = np.argsort(contributions)

    top_positive_idx = sorted_idx[-5:][::-1]
    top_negative_idx = sorted_idx[:5]

    top_positive_words = [feature_names[i] for i in top_positive_idx]
    top_negative_words = [feature_names[i] for i in top_negative_idx]

    return top_positive_words, top_negative_words


# -------------------------------
# Main Flask Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    mode = None
    error = None

    if request.method == "POST":
        try:
            sentiment = request.form.get("sentiment", "positive")
            mode = request.form.get("mode", "manual")

            # -------------------------------
            # Input Handling
            # -------------------------------
            manual_reviews = request.form.get("manual_reviews", "")
            uploaded_file = request.files.get("file")

            if request.form.get("use_sample"):
                if not os.path.isfile(SAMPLE_PATH):
                    raise FileNotFoundError(
                        "Sample dataset is not available."
                    )

                df = pd.read_csv(SAMPLE_PATH)

            elif manual_reviews.strip():
                reviews_list = [
                    review.strip()
                    for review in manual_reviews.splitlines()
                    if review.strip()
                ]
                df = pd.DataFrame({"content": reviews_list})

            elif uploaded_file and uploaded_file.filename:
                if not uploaded_file.filename.lower().endswith(".csv"):
                    raise ValueError("Please upload a CSV file.")

                df = pd.read_csv(uploaded_file)
                df.columns = (
                    df.columns.astype(str).str.strip().str.lower()
                )

            else:
                raise ValueError(
                    "Please try the sample dataset or upload a CSV file."
                )

            # -------------------------------
            # Validate Input
            # -------------------------------
            if "content" not in df.columns:
                raise ValueError(
                    "Invalid CSV format. Your file must contain "
                    "a column named 'content'."
                )

            if df.empty:
                raise ValueError("The dataset contains no reviews.")

            df["content"] = df["content"].fillna("").astype(str)
            df = df[df["content"].str.strip() != ""].copy()

            if df.empty:
                raise ValueError(
                    "No valid reviews were found in the dataset."
                )

            # -------------------------------
            # Prediction
            # -------------------------------
            reviews = df["content"].apply(clean_text)
            X = vectorizer.transform(reviews)

            probs = model.predict_proba(X)

            df["pos_prob"] = probs[:, 1]
            df["neg_prob"] = probs[:, 0]

            # -------------------------------
            # Manual Mode: Top Reviews
            # -------------------------------
            if mode == "manual":

                if sentiment == "positive":
                    filtered = df.sort_values(
                        by="pos_prob", ascending=False
                    )
                else:
                    filtered = df.sort_values(
                        by="neg_prob", ascending=False
                    )

                results = [
                    review[:150] + "..."
                    if len(review) > 150 else review
                    for review in filtered["content"].head(5)
                ]

            # -------------------------------
            # Automatic Mode: Common Reasons
            # -------------------------------
            elif mode == "auto":

                if sentiment == "positive":
                    filtered = df.sort_values(
                        by="pos_prob", ascending=False
                    ).head(3)
                else:
                    filtered = df.sort_values(
                        by="neg_prob", ascending=False
                    ).head(3)

                words = []

                for review in filtered["content"]:
                    vec = vectorizer.transform([clean_text(review)])

                    pos_words, neg_words = get_top_words(vec)

                    if sentiment == "positive":
                        words.extend(pos_words)
                    else:
                        words.extend(neg_words)

                words = [
                    word for word in words
                    if word not in ENGLISH_STOP_WORDS
                    and len(word) > 2
                ]

                mapping = {
                    "amazing": "Strong positive reception",
                    "excellent": "Highly appreciated",
                    "great": "Overall positive experience",
                    "superb": "Exceptional performance",
                    "slow": "Pacing issues",
                    "boring": "Lack of engagement",
                    "confusing": "Storyline issues",
                    "performance": "Acting quality",
                    "performances": "Acting quality",
                    "story": "Story-related feedback",
                    "plot": "Plot-related feedback",
                    "football": "Theme-specific feedback",
                    "player": "Theme-specific feedback",
                    "footage": "Cinematography/visuals"
                }

                mapped_results = list(dict.fromkeys(
                    mapping[word]
                    for word in words
                    if word in mapping
                ))

                results = mapped_results if mapped_results else list(
                    dict.fromkeys(words)
                )[:5]

            else:
                raise ValueError("Invalid analysis mode selected.")

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        results=results,
        mode=mode,
        error=error
    )


# -------------------------------
# Run Locally
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import joblib
import re
import os
import string
import nltk

# -------------------------
# NLTK setup
# -------------------------
nltk.data.path.append("nltk_data")

for resource in ["wordnet", "stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource in ("wordnet", "stopwords") else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir="nltk_data")

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

_ = wordnet.words()  # preload WordNet
lemmatizer = WordNetLemmatizer()

# Preserve negation words — they carry sentiment ("not good" ≠ "good")
_stop_words = set(stopwords.words("english"))
_negation_words = {"not", "no", "never", "nor", "neither", "n't",
                   "without", "nobody", "nothing", "nowhere",
                   "hardly", "barely"}
stop_words = _stop_words - _negation_words

# -------------------------
# CLEANING FUNCTION
# Must match the preprocessing pipeline used during model training
# -------------------------
def clean_text(text):
    text = str(text).lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove URLs and @mentions
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\S+", "user", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize, remove stopwords, lemmatize
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and t.strip()]
    return " ".join(tokens)

# -------------------------
# LOAD MODELS + VECTORIZERS
# Two separate vectorizers — one per model — to match training setup
# -------------------------
try:
    tfidf_binary = joblib.load("tfidf_binary_vectorizer.pkl")
    tfidf_multi  = joblib.load("tfidf_multi_vectorizer.pkl")
    log_reg      = joblib.load("logistic_regression_model.pkl")
    xgb_model    = joblib.load("xgb_multiclass_model.pkl")
    print("All models and vectorizers loaded successfully.")
except Exception as e:
    print("MODEL LOADING ERROR:", str(e))
    tfidf_binary = tfidf_multi = log_reg = xgb_model = None

# -------------------------
# PREDICTION HELPERS
# -------------------------
LABEL_MAP_BINARY     = {0: "Negative", 2: "Positive"}
LABEL_MAP_MULTICLASS = {0: "Negative", 1: "Neutral",  2: "Positive"}

def predict_sentiment_single(tweet, model="logistic"):
    cleaned = clean_text(tweet)

    if model == "logistic":
        vec       = tfidf_binary.transform([cleaned])
        pred_code = log_reg.predict(vec)[0]
        return LABEL_MAP_BINARY.get(int(pred_code), "Unknown")
    else:
        vec       = tfidf_multi.transform([cleaned])
        pred_code = xgb_model.predict(vec)[0]
        return LABEL_MAP_MULTICLASS.get(int(pred_code), "Unknown")

def predict_sentiment_batch(tweets, model="logistic"):
    cleaned = [clean_text(t) for t in tweets]

    if model == "logistic":
        vec        = tfidf_binary.transform(cleaned)
        pred_codes = log_reg.predict(vec)
        label_map  = LABEL_MAP_BINARY
    else:
        vec        = tfidf_multi.transform(cleaned)
        pred_codes = xgb_model.predict(vec)
        label_map  = LABEL_MAP_MULTICLASS

    return [label_map.get(int(code), "Unknown") for code in pred_codes]

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_single():
    """Single-tweet prediction endpoint."""
    data  = request.get_json()
    tweet = data.get("tweet", "").strip()
    model = data.get("model", "logistic")

    if not tweet:
        return jsonify({"error": "No tweet provided."}), 400
    if log_reg is None or xgb_model is None:
        return jsonify({"error": "Models not loaded."}), 500

    sentiment = predict_sentiment_single(tweet, model)
    return jsonify({"tweet": tweet, "predicted_sentiment": sentiment})

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint."""
    data   = request.get_json()
    tweets = data.get("tweets", [])
    model  = data.get("model", "logistic")

    if not tweets:
        return jsonify({"predictions": []})
    if log_reg is None or xgb_model is None:
        return jsonify({"error": "Models not loaded."}), 500

    predictions = predict_sentiment_batch(tweets, model)
    results = [{"tweet": t, "predicted_sentiment": p}
               for t, p in zip(tweets, predictions)]
    return jsonify({"predictions": results})

# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

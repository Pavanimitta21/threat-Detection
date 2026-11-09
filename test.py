import joblib
import re
import nltk
from urllib.parse import urlparse

# Load models
text_model = joblib.load("models/text_model.pkl")
tfidf = joblib.load("models/text_vectorizer.pkl")
url_model = joblib.load("models/url_model.pkl")

# NLTK
nltk.data.path.insert(0, r"C:\Users\NAREN\nltk_data")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))


# =========================
# TEXT TEST
# =========================

def test_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]

    text = " ".join(words)

    x = tfidf.transform([text])

    prediction = text_model.predict(x)[0]
    probability = text_model.predict_proba(x)[0]

    confidence = max(probability) * 100

    print("\nTEXT:", text)
    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 2), "%")


# =========================
# URL FEATURE EXTRACTION
# =========================

def extract_url_features(url):

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.split(":")[0]

    length = len(url)

    digits = len(re.findall(r"\d", url))

    special = len(
        re.findall(r"[^A-Za-z0-9]", url)
    )

    subdomains = max(
        len(domain.split(".")) - 2,
        0
    )

    is_ip = int(
        bool(
            re.match(
                r"^\d{1,3}(\.\d{1,3}){3}$",
                domain
            )
        )
    )

    return [
        length,
        len(domain),
        subdomains,
        digits,
        special,
        url.count("@"),
        url.count("?"),
        url.count("="),
        int(parsed.scheme == "https"),
        is_ip
    ]


# =========================
# URL TEST
# =========================

def test_url(url):

    features = extract_url_features(url)

    prediction = url_model.predict([features])[0]
    probability = url_model.predict_proba([features])[0]

    if prediction == 1:
        result = "Phishing"
        confidence = probability[1] * 100
    else:
        result = "Safe"
        confidence = probability[0] * 100

    print("\nURL:", url)
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2), "%")


# =========================
# TEST CASES
# =========================

test_text(
    "Congratulations! You won a free prize. Click the link now!"
)

test_text(
    "Your meeting is scheduled for tomorrow at 10 AM."
)

test_text(
    "URGENT! Your bank account has been blocked. Verify your details."
)


test_url("https://www.youtube.com")

test_url("https://www.google.com")
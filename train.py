import pandas as pd
import joblib
import re
import nltk
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# NLTK
# =========================

nltk.data.path.insert(0, r"C:\Users\NAREN\nltk_data")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))


# =========================
# TEXT MODEL
# =========================

sms = pd.read_csv("datasets/spam.csv")

sms = sms[["v1", "v2"]]
sms.columns = ["label", "text"]

sms["label"] = sms["label"].map({
    "ham": "safe",
    "spam": "threat"
})


phishing = pd.read_csv(
    "datasets/phishing_dataset_with_category.csv"
)

phishing = phishing[["text", "label"]]
phishing["label"] = "threat"


text_data = pd.concat(
    [
        sms[["text", "label"]],
        phishing[["text", "label"]]
    ],
    ignore_index=True
)


def clean_text(text):

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


text_data["text"] = text_data["text"].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    text_data["text"],
    text_data["label"],
    test_size=0.20,
    random_state=42,
    stratify=text_data["label"]
)


vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    sublinear_tf=True
)


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


text_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


text_model.fit(X_train, y_train)

prediction = text_model.predict(X_test)

print(
    "TEXT MODEL ACCURACY:",
    accuracy_score(y_test, prediction)
)

print(
    classification_report(y_test, prediction)
)


joblib.dump(
    text_model,
    "models/text_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/text_vectorizer.pkl"
)


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


url_data = pd.read_csv(
    "datasets/PhiUSIIL_Phishing_URL_Dataset.csv"
)


# Dataset:
# 0 = phishing
# 1 = legitimate
#
# Convert:
# 1 = phishing
# 0 = safe

y = 1 - url_data["label"]


X = url_data["URL"].apply(
    extract_url_features
).tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


url_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


url_model.fit(X_train, y_train)

prediction = url_model.predict(X_test)


print(
    "URL MODEL ACCURACY:",
    accuracy_score(y_test, prediction)
)

print(
    classification_report(
        y_test,
        prediction,
        target_names=["Safe", "Phishing"]
    )
)


joblib.dump(
    url_model,
    "models/url_model.pkl"
)


print("\nModels trained and saved successfully!")
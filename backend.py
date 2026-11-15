import re
import joblib
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# ============================================================
# NLTK RESOURCES
# ============================================================

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("testing")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    lemmatizer = WordNetLemmatizer()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("Loading trained models...")

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

rf_model = joblib.load(
    "models/random_forest_model.pkl"
)

tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

queue_encoder = joblib.load(
    "models/queue_label_encoder.pkl"
)

priority_encoder = joblib.load(
    "models/priority_label_encoder.pkl"
)

gru_routing_model = load_model(
    "models/gru_routing_model.keras"
)

gru_priority_model = load_model(
    "models/gru_priority_model.keras"
)

print("All trained models loaded successfully!")


# ============================================================
# CONSTANTS
# ============================================================

MAX_LEN = 60


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Preprocess customer support ticket text.

    Steps:
    1. Convert text to lowercase
    2. Remove newline characters
    3. Remove URLs
    4. Remove punctuation and special characters
    5. Remove extra spaces
    6. Remove stopwords
    7. Apply lemmatization
    """

    # Convert input to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace literal newline characters
    text = text.replace("\\n", " ")

    # Remove actual newline characters
    text = text.replace("\n", " ")

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords and perform lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Convert list back to string
    return " ".join(words)


# ============================================================
# RANDOM FOREST ROUTING
# ============================================================

def predict_queue(text):
    """
    Predict the customer support queue using Random Forest.

    Returns:
        predicted_queue
        confidence
    """

    # Preprocess ticket
    clean_text = preprocess_text(text)

    # Convert text to TF-IDF features
    text_tfidf = tfidf.transform(
        [clean_text]
    )

    # Predict queue
    prediction = rf_model.predict(
        text_tfidf
    )[0]

    # Get class probabilities
    probabilities = rf_model.predict_proba(
        text_tfidf
    )[0]

    # Highest probability
    confidence = float(
        np.max(probabilities)
    )

    return prediction, confidence


# ============================================================
# GRU + ATTENTION ROUTING
# ============================================================

def predict_gru_queue(text):
    """
    Predict customer support queue using
    GRU + Attention model.

    Returns:
        predicted_queue
        confidence
    """

    # Preprocess ticket
    clean_text = preprocess_text(text)

    # Convert text into integer sequence
    sequence = tokenizer.texts_to_sequences(
        [clean_text]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Predict probabilities
    probabilities = gru_routing_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Find highest probability class
    predicted_index = int(
        np.argmax(probabilities)
    )

    # Convert numeric class back to queue name
    predicted_queue = queue_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Confidence
    confidence = float(
        probabilities[predicted_index]
    )

    return predicted_queue, confidence


# ============================================================
# GRU + ATTENTION PRIORITY PREDICTION
# ============================================================

def predict_priority(text):
    """
    Predict ticket priority using GRU + Attention.

    Returns:
        predicted_priority
        confidence
        urgency_score
    """

    # Preprocess ticket
    clean_text = preprocess_text(text)

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences(
        [clean_text]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Predict probabilities
    probabilities = priority_model_predict(
        padded_sequence
    )

    # Highest probability class
    predicted_index = int(
        np.argmax(probabilities)
    )

    # Convert numeric class to priority
    predicted_priority = priority_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Confidence
    confidence = float(
        probabilities[predicted_index]
    )

    # Find probability of HIGH priority
    high_index = list(
        priority_encoder.classes_
    ).index("high")

    high_probability = float(
        probabilities[high_index]
    )

    # Urgency score from 0 to 100
    urgency_score = high_probability * 100

    return (
        predicted_priority,
        confidence,
        urgency_score
    )


# ============================================================
# PRIORITY MODEL HELPER
# ============================================================

def priority_model_predict(padded_sequence):
    """
    Generate priority probabilities.
    """

    probabilities = gru_priority_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    return probabilities


# ============================================================
# COMPLETE TICKET PREDICTION
# ============================================================

def predict_ticket(text):
    """
    Complete prediction for one customer ticket.

    Uses:
    - Random Forest for routing
    - GRU + Attention for routing
    - GRU + Attention for priority
    """

    # Random Forest routing
    rf_queue, rf_confidence = predict_queue(
        text
    )

    # GRU routing
    gru_queue, gru_confidence = predict_gru_queue(
        text
    )

    # GRU priority
    priority, priority_confidence, urgency_score = predict_priority(
        text
    )

    return {
        "random_forest_queue": rf_queue,
        "random_forest_confidence": rf_confidence,

        "gru_queue": gru_queue,
        "gru_confidence": gru_confidence,

        "priority": priority,
        "priority_confidence": priority_confidence,

        "urgency_score": urgency_score
    }


# ============================================================
# TEST BACKEND
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("CUSTOMER SUPPORT TICKET ROUTING")
    print("Backend Test")
    print("========================================")

    test_ticket = (
        "I am unable to login to my account. "
        "The system keeps showing an error message."
    )

    print("\nTest Ticket:")
    print(test_ticket)

    print("\nProcessing ticket...")

    result = predict_ticket(
        test_ticket
    )

    print("\n----------------------------------------")
    print("PREDICTION RESULT")
    print("----------------------------------------")

    print(
        "Random Forest Queue:",
        result["random_forest_queue"]
    )

    print(
        "Random Forest Confidence:",
        f"{result['random_forest_confidence'] * 100:.2f}%"
    )

    print(
        "GRU + Attention Queue:",
        result["gru_queue"]
    )

    print(
        "GRU + Attention Confidence:",
        f"{result['gru_confidence'] * 100:.2f}%"
    )

    print(
        "Predicted Priority:",
        result["priority"]
    )

    print(
        "Priority Confidence:",
        f"{result['priority_confidence'] * 100:.2f}%"
    )

    print(
        "Urgency Score:",
        f"{result['urgency_score']:.2f}/100"
    )

    print("\n========================================")
    print("Backend loaded and tested successfully!")
    print("========================================")
import pandas as pd
from transformers import pipeline
from textblob import TextBlob
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk dependencies are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP components
sentiment_pipeline = pipeline("sentiment-analysis")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text.lower())
    return " ".join(
        [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    )


def analyze_claims(df):
    df = df.copy()

    # Preprocess comments
    df["cleaned"] = df["comment"].apply(preprocess)

    # Sentiment
    sentiments = sentiment_pipeline(df["comment"].tolist())
    df["Sentiment"] = [s["label"] for s in sentiments]

    # Polarity
    df["Polarity"] = df["comment"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Topics
    embeddings = sentence_model.encode(df["cleaned"].tolist())
    topics, _ = topic_model.fit_transform(df["cleaned"].tolist(), embeddings)
    df["Topic"] = topics

    # Simple category logic (could be replaced with ML classifier)
    def auto_category(comment):
        comment = comment.lower()
        if "copay" in comment:
            return "Copay Dispute"
        if "duplicate" in comment:
            return "Duplicate Denial"
        if "denied" in comment:
            return "Denied - Policy Not Met"
        if "no payment" in comment:
            return "Payment Missing"
        if "paid lower" in comment or "underpaid" in comment:
            return "Underpayment"
        if "coordination of benefits" in comment or "COB" in comment:
            return "COB Issue"
        return "Other"

    df["Predicted Category"] = df["comment"].apply(auto_category)

    return df

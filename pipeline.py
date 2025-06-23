import os
import re
import nltk
import warnings
import pandas as pd
from transformers import pipeline
from textblob import TextBlob
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")



# Setup environment (no Streamlit here)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Download required nltk data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize NLP tools
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

topic_model = BERTopic(
    embedding_model=sentence_model,
    calculate_probabilities=False,
    verbose=False
)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    )


def analyze_claims(df):
    df = df.copy()

    # Preprocess
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
    
        # Map topic numbers to human-readable labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {
        row["Topic"]: topic_model.get_topic(row["Topic"])
        for idx, row in topic_info.iterrows()
        if row["Topic"] != -1
    }

    def format_topic_label(topic_num):
        if topic_num in topic_labels:
            words = [word for word, _ in topic_labels[topic_num][:4]]
            return f"Topic {topic_num}: " + ", ".join(words)
        return "Miscellaneous"

    df["Topic Label"] = df["Topic"].apply(format_topic_label)


    # Rule-based categories
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

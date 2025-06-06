# 🧠 Healthcare Claims Intelligence Dashboard

## 📋 Overview

This project is a powerful, end-to-end claims analysis system built using Natural Language Processing (NLP) and Machine Learning. It analyzes healthcare insurance feedback to detect patterns, classify issues, extract sentiment, and surface actionable insights.

It features:

* An interactive Streamlit dashboard
* AI-powered claim search (via Gemini & GPT)
* One-click PDF report generation

---

## 🔧 Tech Stack

* **Language:** Python 3.13+
* **Framework:** Streamlit
* **NLP & ML:**

  * HuggingFace Transformers (DistilBERT)
  * Sentence-Transformers
  * BERTopic (for topic modeling)
  * TextBlob (for polarity)
  * NLTK (for preprocessing)
* **Data & Viz:** Pandas, NumPy, Plotly
* **AI Integration:** Google Generative AI (Gemini), OpenAI GPT
* **Reporting:** Jinja2, WeasyPrint

---

## 📁 Folder Structure

```
Health_insurance_claims/
├── app.py               # Streamlit frontend
├── pipeline.py          # NLP pipeline
├── gpt_agent.py         # AI-powered question answering
├── data_utils.py        # Data cleaning & validation
├── reporting.py         # Report generator
├── templates/
│   └── report_template.html
└── requirements.txt     # All dependencies
```

---

## 🧠 Core Components

### 1. `data_utils.py`: Data Handling

* Loads and validates CSV files
* Ensures all required fields (e.g., claim\_id, comment)
* Computes `amount_gap` (expected - paid)

### 2. `pipeline.py`: NLP Engine

* Cleans, tokenizes, and lemmatizes comments
* Sentiment detection via DistilBERT (Positive/Negative)
* Polarity scoring with TextBlob (-1 to +1 scale)
* Topic Modeling with BERTopic (auto clusters)
* Rule-based categorization:

  * Copay Disputes
  * Duplicate Denials
  * Denied Policies
  * COB Issues
  * Payment Gaps
  * Others

### 3. `gpt_agent.py`: Ask Your Claims AI

* Uses semantic search to find related claims
* Answers natural questions like:

  * “Why was this claim denied?”
  * “Which CPT code has the most disputes?”
* Uses Gemini / GPT for contextual response generation

### 4. `reporting.py`: Auto PDF Report Generator

* HTML → PDF via Jinja2 & WeasyPrint
* Includes summaries, insights, and sample claims

### 5. `app.py`: Streamlit Frontend

* Upload claims CSV
* View charts, tables, sentiments, topics
* Ask AI agent
* Download enriched CSV or PDF reports

---

## 📊 Key Features

### ✅ Classification Tags:

* Copay Disputes
* Policy Denials
* Duplicate Denials
* Underpayments / Missing Payments
* COB Issues

### 😊 Sentiment Analysis:

* Quickly detect frustration or positive feedback
* Breakdown by sentiment type
* See polarity scores

### 🧠 Topic Modeling:

* Auto-extract themes from feedback (e.g., “Billing confusion”)
* Group claims by topics using semantic similarity

---

## 🚀 Getting Started

### 1. API Setup

Create a `.env` file with:

```
GOOGLE_API_KEY=your_google_key
OPENAI_API_KEY=your_openai_key  # (optional fallback)
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Sample CSV Format

| claim\_id | comment                                | category                | specialty  | insurance\_type | cpt\_code | amount\_expected | amount\_paid |
| --------- | -------------------------------------- | ----------------------- | ---------- | --------------- | --------- | ---------------- | ------------ |
| 1001      | Denied due to missing documentation... | Denied - Policy Not Met | Cardiology | Plan A          | 99213     | 150.00           | 0.00         |

---

## 📈 How to Use

1. Upload a claims CSV
2. View automated analysis:

   * Sentiment
   * Topics
   * Categorization
3. Ask the AI Agent your questions
4. Download a report with insights

---

## 🔐 Notes

* Requires Python 3.13+
* For PDF reports, install GTK & Cairo
* Large datasets may take time during topic modeling

---

## 🤝 Contributing

PRs welcome! Raise an issue for bugs, ideas, or suggestions.

---

## 📌 Live App

👉 [Streamlit Demo](https://health-insurance-claims-analyzer.streamlit.app/)
👉 [GitHub Repository](https://github.com/kalluru498/health-claims-analyzer)

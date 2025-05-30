# Healthcare Claims Intelligence Dashboard

## 📋 Overview
This project is an advanced healthcare claims analysis system that uses Natural Language Processing (NLP) and Machine Learning to analyze, categorize, and extract insights from healthcare insurance claims data. It features a interactive Streamlit dashboard, automated reporting, and an AI-powered query system.

## 🔧 Technical Stack
- **Python 3.13+**
- **Framework:** Streamlit
- **NLP Libraries:** 
  - Transformers (BERT)
  - Sentence-Transformers
  - BERTopic
  - NLTK
  - TextBlob
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Reporting:** Jinja2, WeasyPrint
- **AI Components:**
  - Google Generative AI (Gemini Model)
  - OpenAI GPT
  - HuggingFace Transformers

## 📁 Project Structure
```
Health_insurance_claims/
├── app.py                 # Main Streamlit application
├── pipeline.py           # NLP processing pipeline
├── gpt_agent.py         # AI query system
├── data_utils.py        # Data loading and preprocessing
├── reporting.py         # Report generation
├── requirements.txt     # Project dependencies
└── templates/          # HTML templates for reports
    └── report_template.html
```

## 🔍 Component Details

### 1. Data Processing (`data_utils.py`)
- Handles CSV file loading and validation
- Performs basic text cleaning and normalization
- Validates required columns: claim_id, comment, category, specialty, etc.
- Calculates payment gaps between expected and paid amounts

### 2. NLP Pipeline (`pipeline.py`)
- **Text Preprocessing:**
  - Tokenization using NLTK
  - Stop word removal
  - Lemmatization
- **Analysis Components:**
  - Sentiment Analysis using Hugging Face Transformers
  - Topic Modeling with BERTopic
  - Custom category classification
  - Polarity scoring using TextBlob

### 3. AI Query System (`gpt_agent.py`)
- Implements semantic search using Sentence Transformers
- Provides natural language interface for querying claims data
- Uses cosine similarity for finding relevant claims
- Integrates multiple AI models:
  - Google Generative AI (Gemini) for advanced reasoning
  - OpenAI GPT for natural language understanding
  - BERT-based models for semantic similarity
- Generates context-aware responses based on similar claims
- Supports multi-model fallback for reliability

### 4. Report Generation (`reporting.py`)
- Generates HTML reports using Jinja2 templates
- Includes summary statistics and visualizations
- Supports custom templating and styling
- Automatically organizes reports by date

### 5. Web Interface (`app.py`)
- Interactive Streamlit dashboard
- Features:
  - CSV file upload and validation
  - Real-time NLP analysis
  - Interactive visualizations
  - AI-powered Q&A system
  - Report generation and download

## 📊 Analysis Features
1. **Category Classification:**
   - Copay Disputes
   - Duplicate Denials
   - Policy-based Denials
   - Payment Issues
   - COB (Coordination of Benefits)
   - Others

2. **Sentiment Analysis:**
   - Positive/Negative classification
   - Polarity scoring
   - Sentiment distribution visualization

3. **Topic Modeling:**
   - Automatic topic extraction
   - Clustering similar claims
   - Topic distribution analysis

## 🚀 Getting Started

1. **API Setup:**
   - Get a Google AI API key from Google Cloud Console
   - Set up your `.env` file with:
     ```
     GOOGLE_API_KEY=your_api_key_here
     OPENAI_API_KEY=your_openai_key_here  # Optional for GPT fallback
     ```

2. **Installation:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Running the Application:**
   ```bash
   streamlit run app.py
   ```

3. **Required Data Format:**
   CSV file with columns:
   - claim_id
   - comment
   - category
   - specialty
   - insurance_type
   - cpt_code
   - amount_expected
   - amount_paid

## 📈 Usage
1. Upload a claims CSV file through the Streamlit interface
2. View automated analysis results:
   - Category distribution
   - Sentiment analysis
   - Topic modeling results
3. Use the AI agent to ask questions about the claims
4. Generate and download detailed reports

## 🔐 Dependencies
See `requirements.txt` for complete list of dependencies.

## 📝 Notes
- The system requires Python 3.13+ due to specific dependency requirements
- For PDF report generation, GTK and Cairo system dependencies are required
- Large datasets may require additional processing time for NLP analysis

## 🤝 Contributing
Feel free to submit issues and enhancement requests!

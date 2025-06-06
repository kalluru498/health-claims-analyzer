import nltk
import os
import warnings
import streamlit as st

def setup_environment():
    """Initialize the environment with required NLTK data and settings"""
    # Suppress warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    # Download NLTK data with progress indicator
    with st.spinner("📚 Setting up NLP components..."):
        resources = {
            'punkt': 'Tokenizer',
            'stopwords': 'Stop words',
            'wordnet': 'WordNet',
            'averaged_perceptron_tagger': 'POS Tagger'
        }
        
        for resource, description in resources.items():
            try:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
                    st.toast(f"✅ Downloaded {description}")
            except Exception as e:
                st.warning(f"⚠️ Could not download {description}: {str(e)}")

    return True

if __name__ == "__main__":
    setup_environment()

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import torch
from pipeline import analyze_claims
from gpt_agent import gpt_response

warnings.filterwarnings("ignore", message=".*torch.classes.*")


# Suppress torch path warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set page config
# Initialize environment
from init_setup import setup_environment
setup_environment()

st.title("📊 Healthcare Claims Intelligence")

# Add GitHub link and documentation
st.markdown("""
    <div style='text-align: right'>
        <small>
            <a href='https://github.com/yourusername/health-claims-analyzer' target='_blank'>📚 Documentation</a>
        </small>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload claims CSV file", type=["csv"], help="Upload a CSV file containing claims data with at least a 'comment' column.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if 'comment' not in df.columns:
        st.error("CSV must contain a 'comment' column.")
    else:
        with st.spinner("🔍 Analyzing comments using NLP and ML..."):
            result_df = analyze_claims(df)

        st.success("✅ Analysis complete!")

        st.write("### 📊 Processed Data")
        st.dataframe(result_df, use_container_width=True)

        st.download_button("📥 Download Analyzed CSV", result_df.to_csv(index=False), file_name="analyzed_claims.csv")

        st.write("### 📁 Category Distribution")
        st.bar_chart(result_df['Predicted Category'].value_counts())

        st.write("### 😊 Sentiment Distribution")
        st.bar_chart(result_df['Sentiment'].value_counts())

        st.write("### 🧠 Top Topics")
        st.table(result_df['Topic'].value_counts().reset_index().rename(columns={'index': 'Topic', 'Topic': 'Count'}).head(5))

    
        st.markdown("## 💬 Ask the AI Agent")
        question = st.text_input("Ask a question about your claims (e.g., 'Why was this denied?')")
        if question:
            with st.spinner("🧠 Thinking..."):
                response = gpt_response(result_df, question)
                st.markdown(f"**📌 Answer:** {response}")


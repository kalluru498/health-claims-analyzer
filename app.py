import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="Claims Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import pandas as pd
import plotly.express as px
import warnings


from pipeline import analyze_claims
from gpt_agent import gpt_response
from init_setup import setup_environment

# --- Environment Setup ---
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
setup_environment()

# --- Page Setup ---
st.title("📊 Healthcare Claims Intelligence")

# --- GitHub Link ---
st.markdown("""
    <div style='text-align: right'>
        <small>
            <a href='https://github.com/kalluru498/health-claims-analyzer' target='_blank'>📚 Documentation</a>
        </small>
    </div>
""", unsafe_allow_html=True)

# --- Show Sample Format First ---
st.subheader("📋 Sample Dataset Format")
sample_df = pd.read_csv("sample_data_set.csv")
st.dataframe(sample_df.head(), use_container_width=True)

st.download_button(
    label="⬇ Download Sample CSV",
    data=sample_df.to_csv(index=False),
    file_name="sample_claims.csv",
    mime="text/csv"
)

st.markdown("---")

# --- File Upload ---
st.subheader("📤 Upload Your Claims CSV")
uploaded_file = st.file_uploader("Upload a CSV file (or use sample above)", type=["csv"])

# --- Data Selection ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.info("✅ Uploaded data loaded.")
else:
    df = sample_df
    st.info("⚠️ No file uploaded. Using sample data for demo.")

# --- Column Check ---
if 'comment' not in df.columns:
    st.error("CSV must contain a 'comment' column.")
else:
    with st.spinner("🔍 Analyzing comments using NLP and ML..."):
        result_df = analyze_claims(df)

    st.success("✅ Analysis complete!")

    # --- Show Results ---
    st.subheader("📄 Processed Data")
    st.dataframe(result_df, use_container_width=True)

    st.download_button("📥 Download Analyzed CSV", result_df.to_csv(index=False), file_name="analyzed_claims.csv")

    # --- Charts ---
    st.subheader("📁 Category Distribution")
    st.bar_chart(result_df['Predicted Category'].value_counts())

    st.subheader("😊 Sentiment Distribution")
    st.bar_chart(result_df['Sentiment'].value_counts())

    st.subheader("🧠 Top Topics")
    st.table(result_df['Topic'].value_counts().reset_index().rename(columns={'index': 'Topic', 'Topic': 'Count'}).head(5))

    # --- AI Agent ---
    st.markdown("## 💬 Ask the AI Agent")
    question = st.text_input("Ask a question about your claims (e.g., 'Why was this denied?')")
    if question:
        with st.spinner("🧠 Thinking..."):
            response = gpt_response(result_df, question)
            st.markdown(f"**📌 Answer:** {response}")

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

from reporting import generate_html_report
from pipeline import analyze_claims
from gpt_agent import gpt_response
from init_setup import setup_environment

# --- Environment Setup ---
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
setup_environment()

# --- Session Defaults ---
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "ai_response" not in st.session_state:
    st.session_state.ai_response = "No question asked yet."
if "dataframe_loaded" not in st.session_state:
    st.session_state.dataframe_loaded = False

# --- Title ---
st.title("📊 Healthcare Claims Intelligence")

# --- GitHub Link ---
st.markdown("""
    <div style='text-align: right'>
        <small>
            <a href='https://github.com/kalluru498/health-claims-analyzer' target='_blank'>📚 Documentation</a>
        </small>
    </div>
""", unsafe_allow_html=True)

# --- Sample Preview ---
st.subheader("📋 Sample Dataset Format")
sample_df = pd.read_csv("sample_data_set.csv")
st.dataframe(sample_df.head(), use_container_width=True)
st.download_button("⬇ Download Sample CSV", data=sample_df.to_csv(index=False), file_name="sample_claims.csv", mime="text/csv")

st.markdown("---")

# --- File Upload ---
st.subheader("📤 Upload Your Claims CSV")
uploaded_file = st.file_uploader("Upload a CSV file (or use sample above)", type=["csv"])

# --- Load Data ---
if uploaded_file and not st.session_state.dataframe_loaded:
    df = pd.read_csv(uploaded_file)
    st.session_state.dataframe_loaded = True
elif not uploaded_file and not st.session_state.dataframe_loaded:
    df = sample_df
    st.session_state.dataframe_loaded = True
else:
    df = sample_df if not uploaded_file else pd.read_csv(uploaded_file)

# --- Comment Column Check ---
if 'comment' not in df.columns:
    st.error("CSV must contain a 'comment' column.")
else:
    if st.session_state.result_df is None:
        with st.spinner("🔍 Analyzing comments using NLP and ML..."):
            st.session_state.result_df = analyze_claims(df)
        st.success("✅ Analysis complete!")

    result_df = st.session_state.result_df

    # --- Output Display ---
    st.subheader("📄 Processed Data")
    st.dataframe(result_df, use_container_width=True)
    st.download_button("📥 Download Analyzed CSV", result_df.to_csv(index=False), file_name="analyzed_claims.csv")

    st.subheader("📁 Category Distribution")
    st.bar_chart(result_df['Predicted Category'].value_counts())

    st.subheader("😊 Sentiment Distribution")
    st.bar_chart(result_df['Sentiment'].value_counts())

    top_topics = (
        result_df['Topic Label']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Topic Description', 'Topic Label': 'Count'})
    )

    st.subheader("🧠 Top Topics")
    st.table(top_topics.head(10))

    # --- AI Agent ---
    st.markdown("## 💬 Ask the AI Agent ")
    question = st.text_input("Ask a question about your claims (e.g., 'Why was this denied?')")

    if question:
        with st.spinner("🧠 Thinking..."):
            ai_result = gpt_response(result_df, question)
            st.session_state.ai_response = ai_result
            st.markdown(f"**📌 Answer:** {ai_result}")

    # --- PDF Report ---
    st.markdown("## 🧾 Generate PDF Report")
    if st.button("📥 Generate PDF Report"):
        with st.spinner("Generating report..."):
            pdf_path = generate_html_report(result_df, ai_summary=st.session_state.ai_response)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download HTML Report", f, file_name="claims_report.html")

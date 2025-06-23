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

# --- Load Data ---
# --- File Mode Switch ---
st.subheader("📂 Select Data Source")
use_sample = st.toggle("🔁 Use Sample Dataset", value=True)

# --- File Upload ---
uploaded_file = None
if not use_sample:
    uploaded_file = st.file_uploader("📤 Upload Your Claims CSV", type=["csv"])

# --- Load Data ---
if use_sample:
    df = pd.read_csv("sample_data_set.csv")
    st.info("ℹ️ Using sample data.")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.info("✅ Uploaded data loaded.")
else:
    df = None
    st.warning("⚠️ Please upload a CSV file to continue.")



# --- Comment Column Check ---
if df is not None:
    if 'comment' not in df.columns:
        st.error("❌ CSV must contain a 'comment' column.")
    else:
        # Run NLP + topic modeling only if not already done for current data
        if st.session_state.result_df is None or st.session_state.df_id != hash(df.to_csv()):
            with st.spinner("🔍 Analyzing comments using NLP and ML..."):
                st.session_state.result_df = analyze_claims(df)
                st.session_state.df_id = hash(df.to_csv())
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
    st.table(top_topics.head(20))

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

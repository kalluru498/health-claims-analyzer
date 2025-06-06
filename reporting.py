import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

def generate_html_report(df: pd.DataFrame, ai_summary: str = ""):
    # Setup Jinja environment
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_template.html")

    # Prepare stats
    total_claims = len(df)

    category_counts = df["Predicted Category"].value_counts()
    sentiment_counts = df["Sentiment"].value_counts()

    category_stats = [
        {
            "name": cat,
            "count": int(count),
            "pct": round((count / total_claims) * 100, 2)
        }
        for cat, count in category_counts.items()
    ]

    sentiment_stats = [
        {
            "label": label,
            "count": int(count),
            "pct": round((count / total_claims) * 100, 2)
        }
        for label, count in sentiment_counts.items()
    ]

    # Basic summary fields
    summary = {
        "total_claims": len(df),
        "top_categories": ', '.join(df['Predicted Category'].value_counts().head(3).index),
        "sentiment_distribution": ', '.join([
            f"{sentiment}: {count}" for sentiment, count in df["Sentiment"].value_counts().items()
        ])
    }

    # Sample data (ensure required fields exist)
    sample_records = df.head(30)[["claim_id", "comment", "Predicted Category", "Sentiment"]]
    samples = [
        {
            "claim_id": row["claim_id"],
            "comment": row["comment"],
            "category": row["Predicted Category"],
            "sentiment": row["Sentiment"]
        }
        for _, row in sample_records.iterrows()
    ]

    # Generate report path
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"reports/claims_report_{timestamp}.html"

    # Render the HTML with the template
    html_content = template.render(
        summary=summary,
        category_stats=category_stats,
        sentiment_stats=sentiment_stats,
        samples=samples,
        ai_agent_summary=ai_summary,
    )

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path

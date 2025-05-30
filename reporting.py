import pandas as pd
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self, template_dir='templates', output_dir='reports'):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template('report_template.html')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, df: pd.DataFrame, filename='claim_report'):
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        full_path = os.path.join(self.output_dir, f"{filename}_{date_str}.html")

        summary = {
            'total_claims': len(df),
            'categories': df['Predicted Category'].value_counts().to_dict(),
            'sentiments': df['Sentiment'].value_counts().to_dict(),
            'top_topics': df['Topic'].value_counts().head(5).to_dict(),
        }

        html_content = self.template.render(
            title="Healthcare Claims NLP Report",
            summary=summary,
            data=df.head(50).to_dict(orient='records'),
            date=date_str
        )

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return full_path

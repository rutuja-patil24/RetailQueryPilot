import gradio as gr
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from backend.inference.query_processor import process_query

# Dummy table and chart
def get_mock_table():
    return pd.DataFrame({
        "Product": ["Electronics", "Clothing"],
        "Sales": [120000, 95000]
    })

def get_mock_chart(intent):
    fig, ax = plt.subplots()
    data = get_mock_table()

    if intent == "compare":
        ax.bar(data["Product"], data["Sales"], color=["blue", "green"])
        ax.set_title("Sales Comparison")
    elif intent == "trend":
        ax.plot(["Q1", "Q2", "Q3", "Q4"], [10000, 15000, 13000, 18000], marker='o')
        ax.set_title("Quarterly Sales Trend")
    elif intent == "anomaly":
        ax.plot(["Week 1", "Week 2", "Week 3", "Week 4"], [1000, 1050, 8000, 1100])
        ax.set_title("Anomaly in Returns")
    else:  # summary or fallback
        ax.pie(data["Sales"], labels=data["Product"], autopct='%1.1f%%')
        ax.set_title("Sales Distribution")

    return fig

# Main function
def handle_query(question):
    result = process_query(question)
    sql = result["sql"]
    intent = result["intent"]
    table = get_mock_table()
    chart = get_mock_chart(intent)
    return sql, intent, table, chart

# Gradio app
demo = gr.Interface(
    fn=handle_query,
    inputs=gr.Textbox(label="Ask your retail question"),
    outputs=[
        gr.Textbox(label="Generated SQL"),
        gr.Textbox(label="Detected Intent"),
        gr.Dataframe(label="Sample Output Table"),
        gr.Plot(label="Suggested Chart")
    ],
    title="🛒 RetailQueryPilot",
    description="Ask retail-specific questions and get SQL + detected intent + chart + data preview!"
)

if __name__ == "__main__":
    demo.launch()

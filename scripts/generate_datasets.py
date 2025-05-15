import pandas as pd

# Dataset 1: NL to SQL
nl_sql_data = [
    {
        "question": "What were the total sales in March?",
        "sql": "SELECT SUM(amount) FROM transactions WHERE MONTH(date) = 3;"
    },
    {
        "question": "Top 5 stores with highest revenue?",
        "sql": "SELECT store_id, SUM(amount) as total FROM transactions GROUP BY store_id ORDER BY total DESC LIMIT 5;"
    },
    {
        "question": "List top 3 products by return rate.",
        "sql": "SELECT product_id, COUNT(*) / (SELECT COUNT(*) FROM transactions) AS return_rate FROM returns GROUP BY product_id ORDER BY return_rate DESC LIMIT 3;"
    },
    {
        "question": "Which category sold the most in Q1?",
        "sql": "SELECT category, SUM(amount) FROM transactions WHERE QUARTER(date) = 1 GROUP BY category ORDER BY SUM(amount) DESC LIMIT 1;"
    },
]

# Dataset 2: Intent classification
intent_data = [
    {"question": "What were the total sales in March?", "intent": "summary"},
    {"question": "Compare sales of electronics vs clothing.", "intent": "compare"},
    {"question": "How did Q1 2024 sales compare to Q1 2023?", "intent": "trend"},
    {"question": "Which product had abnormal returns last week?", "intent": "anomaly"},
    {"question": "Show me a summary of monthly revenue.", "intent": "summary"},
    {"question": "Compare revenue across all store branches.", "intent": "compare"},
]

# Save to CSV
pd.DataFrame(nl_sql_data).to_csv("data/retail_nl_sql_dataset.csv", index=False)
pd.DataFrame(intent_data).to_csv("data/intent_dataset.csv", index=False)

print("✅ Datasets saved to 'data/' folder.")

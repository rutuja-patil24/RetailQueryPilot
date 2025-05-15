# 🛒 RetailQueryPilot

**RetailQueryPilot** is an LLM-powered Business Intelligence assistant designed specifically for retail use cases. It translates natural language questions into SQL queries, detects user intent, and returns interactive insights through charts and tables.

---

## 📌 Key Features

- 🔍 **Natural Language → SQL**
- 🧠 **Intent Detection** (e.g. compare, trend, anomaly)
- 📊 **Data Table + Chart Visualization**
- 🛒 **Retail-Specific Schema**: sales, returns, inventory
- 💡 Powered by: `T5-small`, `DistilBERT`, Gradio, Matplotlib

---

## 🧱 Folder Structure

RetailQueryPilot/
│
├── .github/workflows/ # CI/CD pipelines
├── backend/
│ ├── inference/ # query_processor.py (inference engine)
│ └── model_training/
│ ├── nl_to_sql_t5/ # T5 model training
│ └── intent_classifier_bert/ # DistilBERT intent classifier
├── data/ # retail_nl_sql_dataset.csv + intent_dataset.csv
├── frontend/ # Gradio UI (app.py)
├── scripts/ # generate_datasets.py
├── requirements.txt
└── .gitignore


---

## 🧪 How to Run

### 🔧 Setup

```bash
pip install -r requirements.txt
```
### 🏁 Train Both Models
# Train NL → SQL (T5)
```bash
python backend/model_training/nl_to_sql_t5/train.py
```
# Train intent classifier (DistilBERT)
```bash
python backend/model_training/intent_classifier_bert/train.py
```
###⚡ Run Inference (backend logic)

```bash
python backend/inference/query_processor.py
```
### 🎨 Launch the UI (Gradio app)

```bash
python frontend/app.py
```

### Then open: http://127.0.0.1:7860

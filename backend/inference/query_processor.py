from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

# Load SQL model
sql_model_path = "backend/model_training/nl_to_sql_t5/final_model"
sql_tokenizer = T5Tokenizer.from_pretrained(sql_model_path)
sql_model = T5ForConditionalGeneration.from_pretrained(sql_model_path)

# Load Intent model
intent_model_path = "backend/model_training/intent_classifier_bert/final_model"
intent_tokenizer = DistilBertTokenizerFast.from_pretrained(intent_model_path)
intent_model = DistilBertForSequenceClassification.from_pretrained(intent_model_path)

# Load label map
label_map_file = "backend/model_training/intent_classifier_bert/label_map.txt"
id2label = {}
with open(label_map_file, "r") as f:
    for line in f:
        id, label = line.strip().split()
        id2label[int(id)] = label

# Predict SQL
def generate_sql(nl_question):
    input_text = "Translate English to SQL: " + nl_question
    input_ids = sql_tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = sql_model.generate(
    input_ids,
    max_length=128,
    num_beams=4,
    early_stopping=True
)

    return sql_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Predict intent
def detect_intent(nl_question):
    tokens = intent_tokenizer(nl_question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = intent_model(**tokens).logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return id2label[predicted_class_id]

# Combined
def process_query(nl_question):
    sql = generate_sql(nl_question)
    intent = detect_intent(nl_question)
    return {
        "question": nl_question,
        "sql": sql,
        "intent": intent
    }

# Example
if __name__ == "__main__":
    question = "Compare sales of electronics vs clothing."
    result = process_query(question)
    print("🧠 Detected Intent:", result["intent"])
    print("📝 Generated SQL:", result["sql"])

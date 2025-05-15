import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# Load data
df = pd.read_csv("data/intent_dataset.csv")
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])  # e.g., summary = 2, etc.

# Save label mapping
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
with open("backend/model_training/intent_classifier_bert/label_map.txt", "w") as f:
    for label, id in label_map.items():
        f.write(f"{id} {label}\n")

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(example["question"], truncation=True)

dataset = Dataset.from_pandas(df[["question", "label"]])
tokenized = dataset.map(preprocess, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# Training args
training_args = TrainingArguments(
    output_dir="./backend/model_training/intent_classifier_bert/checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    logging_steps=5,
    save_steps=5,
    evaluation_strategy="no",
    save_total_limit=2,
    overwrite_output_dir=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train
trainer.train()

# Save model
model.save_pretrained("backend/model_training/intent_classifier_bert/final_model")
tokenizer.save_pretrained("backend/model_training/intent_classifier_bert/final_model")

print("✅ Intent classification model trained and saved.")

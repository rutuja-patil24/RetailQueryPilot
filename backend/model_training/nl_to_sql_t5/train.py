import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv("data/retail_nl_sql_dataset.csv")
df['input'] = "Translate English to SQL: " + df['question']
df = df[['input', 'sql']].rename(columns={"sql": "output"})

# Tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize
def tokenize(batch):
    input_enc = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=128)
    output_enc = tokenizer(batch["output"], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc

# Dataset
hf_dataset = Dataset.from_pandas(df)
tokenized_dataset = hf_dataset.map(tokenize, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./backend/model_training/nl_to_sql_t5/checkpoints",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    overwrite_output_dir=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./backend/model_training/nl_to_sql_t5/final_model")
tokenizer.save_pretrained("./backend/model_training/nl_to_sql_t5/final_model")

print("✅ Model trained and saved to final_model/")

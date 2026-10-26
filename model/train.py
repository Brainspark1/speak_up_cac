import os

from transformers import DataCollatorForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from compute_metrics import compute_metrics

# defining tokenizer, data_collator, importing in dataset

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
ds = load_from_disk('../dataset/mario_token_classification_tokenized')
split_ds = ds.train_test_split(test_size=0.2, seed=42)

#providing label2id and id2label for model
label2id = {
    "O": 0,
    "B-ACTION": 1,
    "I-ACTION": 2,
    "B-TARGET": 3,
    "I-TARGET": 4,
    "B-CORRECTION": 5,
    "I-CORRECTION": 6,
}
id2label = {
    0: "O",
    1: "B-ACTION",
    2: "I-ACTION",
    3: "B-TARGET",
    4: "I-TARGET",
    5: "B-CORRECTION",
    6: "I-CORRECTION",
}

# initializing huggingface model and trainer
model = AutoModelForTokenClassification.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="GAMEBERT",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= split_ds["train"],
    eval_dataset= split_ds["test"],
    processing_class=tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("./token_classifier")
trainer.push_to_hub('Saggarwal/GAMEBERT')
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
import torch

# === 1. Dataset WikiANN in italiano ===
dataset = load_dataset("wikiann", "it")
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# === 2. Configurazione dinamica (GPU 4 GB vs 8 GB) ===
if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram <= 4.1:
        print("🔧 Config: Laptop (4 GB VRAM)")
        batch_size = 1
        max_length = 128
    else:
        print("🔧 Config: Desktop (8 GB VRAM)")
        batch_size = 4
        max_length = 256
else:
    raise RuntimeError("❌ No GPU detected.")

# === 3. Tokenizer & Modello: UmBERTo ===
model_name = "Musixmatch/umberto-commoncrawl-cased-v1"
model_dir = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# === 4. Tokenizzazione e allineamento ===
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=max_length
    )
    word_ids = tokenized_inputs.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(example["ner_tags"][word_idx])
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 🔬 (Facoltativo: rimuovi per full training)
dataset["train"] = dataset["train"].select(range(200))
dataset["validation"] = dataset["validation"].select(range(50))

tokenized_train = dataset["train"].map(tokenize_and_align_labels, batched=False)
tokenized_val = dataset["validation"].map(tokenize_and_align_labels, batched=False)

# === 5. Data collator e metriche ===
data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

# === 6. TrainingArguments ===
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    fp16=True
)

# === 7. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === 8. Train & save ===
trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

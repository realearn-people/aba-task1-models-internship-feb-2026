
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ----------------------------
# 0) CONFIG
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"
MODEL_NAME = "t5-base"

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 4

SEED = 42
TEST_SIZE = 0.2

LABELS = ["negative", "positive"]
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "Negative", 1: "Positive"}

# ----------------------------
# 1) Prompt (หัวใจของ T5)
# ----------------------------
def build_prompt(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    return f"sentiment classification (negative, positive): {text}"

def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(c for c in s if c.isalpha())
    if s.startswith("neg"):
        return "negative"
    if s.startswith("pos"):
        return "positive"
    return "negative"

def labels_to_ids(texts):
    return np.array([LABEL2ID[normalize_label(t)] for t in texts])

# ----------------------------
# 2) Load + clean data
# ----------------------------
df = pd.read_excel(EXCEL_PATH)

df = df[[df.columns[0], df.columns[6], df.columns[7]]]
df.columns = ["id", "text", "sentiment"]

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

df = df[df["sentiment"].isin(LABELS)].copy()
df["label"] = df["sentiment"].map(LABEL2ID)

print("Label counts:")
print(df["sentiment"].value_counts())

# ----------------------------
# 3) Train / Test split
# ----------------------------
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["label"]
)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ----------------------------
# 4) Tokenization
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    inputs = [build_prompt(t) for t in batch["text"]]
    targets = batch["sentiment"]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    labels = tokenizer(
        text_target=targets,
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = Dataset.from_pandas(train_df[["text", "sentiment"]]).map(preprocess, batched=True)
test_ds  = Dataset.from_pandas(test_df[["text", "sentiment"]]).map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# ----------------------------
# 5) Model
# ----------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ----------------------------
# 6) Metrics
# ----------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    y_pred = labels_to_ids(decoded_preds)
    y_true = labels_to_ids(decoded_labels)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": p,
        "recall_macro": r,
    }

# ----------------------------
# 7) Training arguments (ประหยัด disk)
# ----------------------------
args = Seq2SeqTrainingArguments(
    output_dir="t5_binary_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,            # ⭐ สำคัญมาก
    learning_rate=3e-4,            # T5 ใช้ LR สูงกว่า BERT
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ----------------------------
# 8) Train
# ----------------------------
start = time.time()
trainer.train()
train_time = time.time() - start

print(f"\nTraining time: {timedelta(seconds=int(train_time))}")

# ----------------------------
# 9) Final evaluation
# ----------------------------
pred = trainer.predict(test_ds)

decoded_preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(
    np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id),
    skip_special_tokens=True
)

y_pred = labels_to_ids(decoded_preds)
y_true = labels_to_ids(decoded_labels)

print("\n=== TEST REPORT ===")
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"], digits=4))

# ----------------------------
# 10) Save predictions to Excel
# ----------------------------
out_df = test_df.copy().reset_index(drop=True)

out_df["pred_text_raw"] = decoded_preds
out_df["pred_sentiment"] = [ID2LABEL[i] for i in y_pred]
out_df["correct"] = (y_pred == y_true)

OUTPUT_XLSX = "t5_binary_predictions.xlsx"
out_df.to_excel(OUTPUT_XLSX, index=False)

print(f"\nSaved prediction results to: {OUTPUT_XLSX}")


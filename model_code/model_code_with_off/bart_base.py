#BERT-base-uncase version
#3 sentiment - pos,neg,off
import numpy as np
import pandas as pd
import torch
import time
from datetime import timedelta


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix


from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------
# 0) Config
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/Original ABA Dataset for Version 2 (Oct 23, 2025), Senior Project, MUICT.xlsx"
MODEL_NAME = "facebook/bart-base"

MAX_LENGTH = 256
TRAIN_RATIO = 0.8
SEED = 42

# Columns by position: A=id, E=text, H=sentiment
ID_COL_POS = 0     # Column A
TEXT_COL_POS = 6   # Column G
SENT_COL_POS = 7   # Column H

# 3-class mapping
label2id = {"negative": 0, "off": 1, "positive": 2}
id2label = {0: "Negative", 1: "Off", 2: "Positive"}

# ----------------------------
# 1) Load Excel and select columns A/E/H
# ----------------------------
df_raw = pd.read_excel(EXCEL_PATH)

if df_raw.shape[1] <= max(ID_COL_POS, TEXT_COL_POS, SENT_COL_POS):
    raise ValueError(
        f"Excel has {df_raw.shape[1]} columns; need at least {SENT_COL_POS+1} columns for A/E/H."
    )

id_col = df_raw.columns[ID_COL_POS]
text_col = df_raw.columns[TEXT_COL_POS]
sent_col = df_raw.columns[SENT_COL_POS]

df = df_raw[[id_col, text_col, sent_col]].rename(
    columns={id_col: "id", text_col: "text", sent_col: "sentiment"}
).copy()

# Clean text
df = df.dropna(subset=["id", "text"]).copy()
df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
df = df[df["text"] != ""].copy()

# Clean sentiment (map blank/NaN -> off)
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df.loc[df["sentiment"].isin(["", "nan", "none"]), "sentiment"] = "off"

# Keep only 3 classes
df["label"] = df["sentiment"].map(label2id)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

print("Rows:", len(df))
print("Label counts:\n", df["sentiment"].value_counts())

# ----------------------------
# 2) Split train/val/test (stratified)
# ----------------------------
train_df, temp_df = train_test_split(
    df, test_size=(1 - TRAIN_RATIO), random_state=SEED, stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
)

print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

# ----------------------------
# 3) Tokenization
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)
val_ds   = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)
test_ds  = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# 4) Model (BERT base uncased, 3 labels)
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# ----------------------------
# 5) Metrics (macro F1 recommended for 3-class)
# ----------------------------
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
    }

# ----------------------------
# 6) Training
# ----------------------------
args = TrainingArguments(
    output_dir="bart_3class_out",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ----------------------------
# Timing: Training
# ----------------------------
train_start_time = time.time()

trainer.train()

train_end_time = time.time()
train_seconds = train_end_time - train_start_time
print(f"\nTraining time: {timedelta(seconds=int(train_seconds))}")

eval_start_time = time.time()

# ----------------------------
# 7) Evaluate on test
# ----------------------------
test_metrics = trainer.evaluate(test_ds)
print("\nTEST METRICS:", test_metrics)

# Get predictions on test set
pred_output = trainer.predict(test_ds)

# pred_output.predictions may be tuple in some models -> safe handling
preds_raw = pred_output.predictions
if isinstance(preds_raw, tuple):
    preds_raw = preds_raw[0]

y_pred = np.argmax(preds_raw, axis=1)
y_true = pred_output.label_ids

# Overall metrics (macro + weighted)
acc = accuracy_score(y_true, y_pred)
p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

print("\n=== TEST (Overall) ===")
print(f"Accuracy        : {acc:.4f}")
print(f"Precision (macro): {p_macro:.4f}")
print(f"Recall (macro)   : {r_macro:.4f}")
print(f"F1 (macro)       : {f_macro:.4f}")
print(f"Precision (w)    : {p_weighted:.4f}")
print(f"Recall (w)       : {r_weighted:.4f}")
print(f"F1 (w)           : {f_weighted:.4f}")

# Per-class report
target_names = [id2label[0], id2label[1], id2label[2]]
print("\n=== TEST (Per-class report) ===")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(cm)

eval_end_time = time.time()
eval_seconds = eval_end_time - eval_start_time
print(f"Evaluation time: {timedelta(seconds=int(eval_seconds))}")

# ----------------------------
# 8) Save model
# ----------------------------
SAVE_DIR = "bart_base_sentiment_3class_model"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nSaved model to: {SAVE_DIR}")

# ----------------------------
# 9) Predict on ALL rows + export (id, text, gold, pred)
# ----------------------------
best_model = trainer.model
best_model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
best_model.to(device)

def predict_texts(texts, batch_size=64):
    preds_all = []
    probs_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = best_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

        preds_all.extend(preds.tolist())
        probs_all.extend(probs.tolist())

    return np.array(preds_all), np.array(probs_all)

# ----------------------------
# Timing: Inference (predict all rows)
# ----------------------------
infer_start_time = time.time()

all_texts = df["text"].tolist()
pred_ids, probs = predict_texts(all_texts)

infer_end_time = time.time()
infer_seconds = infer_end_time - infer_start_time
print(f"Inference time: {timedelta(seconds=int(infer_seconds))}")
print(f"Avg inference time per sample: {infer_seconds/len(all_texts):.6f} sec")


out_df = df[["id", "text", "sentiment"]].copy()
out_df["pred_id"] = pred_ids
out_df["pred_sentiment"] = out_df["pred_id"].map(id2label)

out_df["prob_negative"] = probs[:, 0]
out_df["prob_off"] = probs[:, 1]
out_df["prob_positive"] = probs[:, 2]

OUTPUT_XLSX = "bart_base(4).xlsx"
OUTPUT_CSV = "bart_base(4).csv"
out_df.to_excel(OUTPUT_XLSX, index=False)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved predictions to: {OUTPUT_XLSX}")
print(f"Saved predictions to: {OUTPUT_CSV}")

# Quick demo
demo = "New, comfortable apartments"
demo_pred, demo_probs = predict_texts([demo])
print("\nDEMO:", demo, "->", id2label[int(demo_pred[0])], "probs=", demo_probs[0].tolist())

# ----------------------------
# Save runtime summary to CSV
# ----------------------------
runtime_df = pd.DataFrame({
    "stage": ["training", "evaluation", "inference"],
    "seconds": [train_seconds, eval_seconds, infer_seconds],
})

runtime_df["time_hms"] = runtime_df["seconds"].apply(
    lambda x: str(timedelta(seconds=int(x)))
)

RUNTIME_CSV = "runtime_bart_base_summary.csv"
runtime_df.to_csv(RUNTIME_CSV, index=False)
print(f"\nSaved runtime summary to: {RUNTIME_CSV}")
print(runtime_df)


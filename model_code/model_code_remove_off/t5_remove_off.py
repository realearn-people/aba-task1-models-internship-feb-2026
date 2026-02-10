# T5-base (seq2seq text-to-text) for 3-class sentiment: negative / off / positive
# Excel columns: A=id, G=text, H=sentiment

import numpy as np
import pandas as pd
import torch
import time
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ----------------------------
# 0) Config
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"
MODEL_NAME = "t5-base"   # ต้องเป็นตัวเล็ก

TRAIN_RATIO = 0.8
SEED = 42

# Columns by position: A=id, G=text, H=sentiment
ID_COL_POS = 0     # Column A
TEXT_COL_POS = 6   # Column G
SENT_COL_POS = 7   # Column H

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 4

# 3-class mapping
label_list = ["negative","positive"]
label2id = {"negative": 0,"positive": 1}
id2label = {0: "Negative",1: "Positive"}

# ----------------------------
# Prompt (สำคัญสำหรับ T5)
# ----------------------------
def build_prompt(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    return f"sentiment classification (negative, positive): {text}"

def normalize_pred_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(ch for ch in s if ch.isalpha())
    if s.startswith("neg"):
        return "negative"
    if s.startswith("pos"):
        return "positive"
    # fallback
    return "off"

def preds_to_ids(decoded_preds):
    norm = [normalize_pred_label(x) for x in decoded_preds]
    return np.array([label2id.get(x, 1) for x in norm])  # default off

# ----------------------------
# 1) Load Excel
# ----------------------------
df_raw = pd.read_excel(EXCEL_PATH)

if df_raw.shape[1] <= max(ID_COL_POS, TEXT_COL_POS, SENT_COL_POS):
    raise ValueError(
        f"Excel has {df_raw.shape[1]} columns; need at least {SENT_COL_POS+1} columns for A/G/H."
    )

id_col = df_raw.columns[ID_COL_POS]
text_col = df_raw.columns[TEXT_COL_POS]
sent_col = df_raw.columns[SENT_COL_POS]

df = df_raw[[id_col, text_col, sent_col]].rename(
    columns={id_col: "id", text_col: "text", sent_col: "sentiment"}
).copy()

df = df.dropna(subset=["id", "text"]).copy()
df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
df = df[df["text"] != ""].copy()

df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df.loc[df["sentiment"].isin(["", "nan", "none"]), "sentiment"] = "off"
df = df[df["sentiment"].isin(label_list)].copy()

df["label"] = df["sentiment"].map(label2id).astype(int)

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
# 3) Tokenization (seq2seq)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    inputs = [build_prompt(t) for t in batch["text"]]
    targets = [str(s).strip().lower() for s in batch["sentiment"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False,
    )

    # ✅ tokenize target/label แบบใหม่
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_ds = Dataset.from_pandas(train_df[["id", "text", "sentiment"]].reset_index(drop=True)).map(preprocess, batched=True)
val_ds   = Dataset.from_pandas(val_df[["id", "text", "sentiment"]].reset_index(drop=True)).map(preprocess, batched=True)
test_ds  = Dataset.from_pandas(test_df[["id", "text", "sentiment"]].reset_index(drop=True)).map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)

# ----------------------------
# 4) Model
# ----------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ----------------------------
# 5) Metrics (decode generated text)
# ----------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    y_pred = preds_to_ids(decoded_preds)
    y_true = preds_to_ids(decoded_labels)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
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
args = Seq2SeqTrainingArguments(
    output_dir="t5_baseremove_off_3class_out",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,               # ✅ T5 มักใช้สูงกว่า BERT
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    logging_steps=50,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    #tokenizer=tokenizer,           # ตัวนี้จะทำงานได้เมื่อ transformers >= 4.20.0
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

# ----------------------------
# 7) Evaluate on test + detailed report
# ----------------------------
eval_start_time = time.time()
test_metrics = trainer.evaluate(test_ds)
print("\nTEST METRICS:", test_metrics)

pred_output = trainer.predict(test_ds)
preds = pred_output.predictions
labels = pred_output.label_ids

decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

y_pred = preds_to_ids(decoded_preds)
y_true = preds_to_ids(decoded_labels)

acc = accuracy_score(y_true, y_pred)
p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

print("\n=== TEST (Overall) ===")
print(f"Accuracy        : {acc:.4f}")
print(f"Precision (macro): {p_macro:.4f}")
print(f"Recall (macro)   : {r_macro:.4f}")
print(f"F1 (macro)       : {f_macro:.4f}")
print(f"Precision (w)    : {p_w:.4f}")
print(f"Recall (w)       : {r_w:.4f}")
print(f"F1 (w)           : {f_w:.4f}")

target_names = [id2label[0], id2label[1]]
print("\n=== TEST (Per-class report) ===")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(cm)

eval_end_time = time.time()
eval_seconds = eval_end_time - eval_start_time
print(f"Evaluation time: {timedelta(seconds=int(eval_seconds))}")

# ----------------------------
# 8) Save model
# ----------------------------
SAVE_DIR = "t5_remove_off_sentiment_3class_model"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nSaved model to: {SAVE_DIR}")

# ----------------------------
# 9) Predict on ALL rows + export CSV/XLSX
# ----------------------------
def predict_texts_text2text(texts, batch_size=32):
    preds_text = []
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    trainer.model.to(device)

    for i in range(0, len(texts), batch_size):
        batch_prompts = [build_prompt(t) for t in texts[i:i+batch_size]]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen = trainer.model.generate(**enc, max_length=MAX_TARGET_LENGTH)

        out = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds_text.extend(out)

    return preds_text

infer_start_time = time.time()
all_texts = df["text"].tolist()
pred_texts = predict_texts_text2text(all_texts, batch_size=32)
infer_end_time = time.time()
infer_seconds = infer_end_time - infer_start_time

pred_norm = [normalize_pred_label(x) for x in pred_texts]
pred_ids = np.array([label2id.get(x, 1) for x in pred_norm])

print(f"Inference time: {timedelta(seconds=int(infer_seconds))}")
print(f"Avg inference time per sample: {infer_seconds/len(all_texts):.6f} sec")

out_df = df[["id", "text", "sentiment"]].copy()
out_df["pred_text_raw"] = pred_texts
out_df["pred_sentiment"] = [id2label[i] for i in pred_ids]

OUTPUT_XLSX = "T5-base_remove_off.xlsx"
OUTPUT_CSV = "T5-base_remove_off.csv"
out_df.to_excel(OUTPUT_XLSX, index=False)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved predictions to: {OUTPUT_XLSX}")
print(f"Saved predictions to: {OUTPUT_CSV}")

# Demo
demo = "New, comfortable apartments"
demo_out = predict_texts_text2text([demo], batch_size=1)[0]
print("\nDEMO:", demo, "->", normalize_pred_label(demo_out), "| raw:", demo_out)

# ----------------------------
# 10) Save runtime summary
# ----------------------------
runtime_df = pd.DataFrame({
    "stage": ["training", "evaluation", "inference"],
    "seconds": [train_seconds, eval_seconds, infer_seconds],
})
runtime_df["time_hms"] = runtime_df["seconds"].apply(lambda x: str(timedelta(seconds=int(x))))

RUNTIME_CSV = "runtime_T5-base_remove_off_summary.csv"
runtime_df.to_csv(RUNTIME_CSV, index=False)
print(f"\nSaved runtime summary to: {RUNTIME_CSV}")
print(runtime_df)

# autotune_binary.py
# Auto Fine-tune (Hyperparameter Search) POS/NEG with Hugging Face Trainer + Optuna
# Excel columns: A=id, G=text, H=sentiment

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# ----------------------------
# 0) CONFIG
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"
MODEL_NAME = "bert-base-uncased"  

MAX_LENGTH = 256
SEED = 42

TEST_SIZE = 0.2
VAL_SIZE_IN_TRAINVAL = 0.2

N_TRIALS = 3
OUTPUT_DIR = "bert_autotune_outputs"

# Excel columns by index
ID_COL_POS = 0
TEXT_COL_POS = 6
SENT_COL_POS = 7

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "Negative", 1: "Positive"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1) Load + clean
# ----------------------------
df_raw = pd.read_excel(EXCEL_PATH)

id_col = df_raw.columns[ID_COL_POS]
text_col = df_raw.columns[TEXT_COL_POS]
sent_col = df_raw.columns[SENT_COL_POS]

df = df_raw[[id_col, text_col, sent_col]].rename(
    columns={id_col: "id", text_col: "text", sent_col: "sentiment"}
).copy()

df = df.dropna(subset=["text", "sentiment"]).copy()
df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

# keep only pos/neg
df = df[df["sentiment"].isin(LABEL2ID.keys())].copy()
df["label"] = df["sentiment"].map(LABEL2ID).astype(int)
df = df.reset_index(drop=True)

print("Rows:", len(df))
print("Label counts:\n", df["sentiment"].value_counts())

# ----------------------------
# 2) Split Train/Val/Test
# ----------------------------
trainval_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
)
train_df, val_df = train_test_split(
    trainval_df, test_size=VAL_SIZE_IN_TRAINVAL, random_state=SEED, stratify=trainval_df["label"]
)

print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

# ----------------------------
# 3) Tokenize + Dataset
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

def make_ds(frame: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(frame[["text", "label"]].reset_index(drop=True))
    return ds.map(tokenize_fn, batched=True)

train_ds = make_ds(train_df)
val_ds   = make_ds(val_df)
test_ds  = make_ds(test_df)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# 4) Metrics
# ----------------------------
def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(preds, tuple):
        preds = preds[0]

    y_pred = np.argmax(preds, axis=-1)
    y_true = labels

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
# 5) model_init (จำเป็นสำหรับ HPO)
# ----------------------------
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ----------------------------
# 6) Trainer base
# ----------------------------
base_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "bert_hpo_runs"),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,           
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    args=base_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    model_init=model_init,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], #ให้หยุดการ train อัตโนมัติ ถ้าผลไม่ดีขึ้น
)

# ----------------------------
# 7) Hyperparameter search (Auto)
# ----------------------------
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [2, 3, 4]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=N_TRIALS,
)

print("\n=== BEST HYPERPARAMETERS ===")
print(best_run.hyperparameters)

# ----------------------------
# 8) Train final model with best hyperparameters เอาโมเดลที่ดีที่สุดมา train อีกรอบ
# ----------------------------
final_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "bert_best_model_run"),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
    **best_run.hyperparameters
)

final_trainer = Trainer(
    model=model_init(),
    args=final_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

final_trainer.train()

# ----------------------------
# 9) Evaluate on test + report
# ----------------------------
test_metrics = final_trainer.evaluate(test_ds)
print("\nTEST METRICS:", test_metrics)

pred_output = final_trainer.predict(test_ds)
preds = pred_output.predictions
if isinstance(preds, tuple):
    preds = preds[0]

y_pred = np.argmax(preds, axis=-1)
y_true = pred_output.label_ids

print("\n=== TEST classification_report ===")
print(classification_report(y_true, y_pred, target_names=[ID2LABEL[0], ID2LABEL[1]], digits=4))

print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(confusion_matrix(y_true, y_pred))

# ----------------------------
# 10) Save final best model
# ----------------------------
SAVE_DIR = os.path.join(OUTPUT_DIR, "bert_best_model_saved")
final_trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("\nSaved best model to:", SAVE_DIR)


import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------
# 0) CONFIG
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"
MODEL_NAME = "bert-base-uncased"  

MAX_LENGTH = 256
SEED = 42
TEST_SIZE = 0.2
K_LIST = [1, 3]

# Columns by position: A=id, G=text, H=sentiment
ID_COL_POS = 0     # Column A
TEXT_COL_POS = 6   # Column G
SENT_COL_POS = 7   # Column H

# Binary mapping
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "Negative", 1: "Positive"}


# ----------------------------
# 1) Load + clean data
# ----------------------------
def load_dataset_from_excel(path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(path)

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

    # clean text
    df = df.dropna(subset=["id", "text"]).copy()
    df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    df = df[df["text"] != ""].copy()

    # clean sentiment
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # keep only pos/neg
    df = df[df["sentiment"].isin(LABEL2ID.keys())].copy()

    df["label"] = df["sentiment"].map(LABEL2ID).astype(int)
    df = df.reset_index(drop=True)
    return df


# ----------------------------
# 2) Tokenize + HF Dataset
# ----------------------------
def make_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    hf = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return hf.map(tokenize_fn, batched=True)


# ----------------------------
# 3) Metrics
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
# 4) Train one fold
# ----------------------------
def train_one_fold(train_df, val_df, test_df, fold_name: str): #ใช้สำหรับ train + evaluate โมเดลใน 1 fold
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = make_hf_dataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = make_hf_dataset(val_df, tokenizer, MAX_LENGTH)
    test_ds = make_hf_dataset(test_df, tokenizer, MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    out_dir = os.path.join("bert_kfold_outputs", fold_name)
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, #กันเมมเต็ม
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # timing
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    t1 = time.time()
    test_metrics = trainer.evaluate(test_ds)
    test_time = time.time() - t1

    test_metrics["train_seconds"] = train_time
    test_metrics["test_seconds"] = test_time

    return test_metrics


# ----------------------------
# 5) K-fold runner
# ----------------------------
def run_kfold(df: pd.DataFrame): #คุมการทดลองทั้งหมด
    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE, #แยก test set เอาไว้สำหรับทุกการทดลอง (test set = 20)
        random_state=SEED,
        stratify=df["label"]
    )

    print(f"\nTotal rows: {len(df)}")
    print(f"TrainVal rows: {len(trainval_df)}")
    print(f"Test rows: {len(test_df)}")

    all_results = []

    for K in K_LIST: #วนลูปทดลองตามค่า K ใน K_LIST
        print("\n" + "=" * 60)
        print(f"RUN K = {K}")
        print("=" * 60)

        if K == 1:
            train_df, val_df = train_test_split(
                trainval_df,
                test_size=0.2,
                random_state=SEED,
                stratify=trainval_df["label"]
            )

            print(f"K=1 | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

            metrics = train_one_fold(train_df, val_df, test_df, fold_name="K1_single")

            all_results.append({
                "K": 1,
                "fold": 1,
                "train_n": len(train_df),
                "val_n": len(val_df),
                "test_n": len(test_df),
                **metrics
            })
            continue

        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED) # กรณี K > 1
        X = trainval_df["text"].values
        y = trainval_df["label"].values

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            fold_train_df = trainval_df.iloc[train_idx].copy()
            fold_val_df = trainval_df.iloc[val_idx].copy()

            print(f"Fold {fold}/{K} | Train: {len(fold_train_df)} | Val: {len(fold_val_df)} | Test: {len(test_df)}")

            metrics = train_one_fold(
                fold_train_df,
                fold_val_df,
                test_df,
                fold_name=f"K{K}_fold{fold}"
            )

            all_results.append({
                "K": K,
                "fold": fold,
                "train_n": len(fold_train_df),
                "val_n": len(fold_val_df),
                "test_n": len(test_df),
                **metrics
            })

    results_df = pd.DataFrame(all_results)

    summary_df = (
        results_df
        .groupby("K")[["eval_accuracy", "eval_f1_macro", "eval_precision_macro", "eval_recall_macro", "train_seconds", "test_seconds"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    return results_df, summary_df


# ----------------------------
# 6) MAIN
# ----------------------------
if __name__ == "__main__": 
    df = load_dataset_from_excel(EXCEL_PATH)

    print("Label counts:")
    print(df["sentiment"].value_counts())

    results_df, summary_df = run_kfold(df)

    os.makedirs("bert_kfold_outputs", exist_ok=True)
    results_csv = os.path.join("bert_kfold_outputs", "bert_kfold_result.csv")
    summary_csv = os.path.join("bert_kfold_outputs", "bert_kfold_summary.csv")

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(" -", results_csv)
    print(" -", summary_csv)

    print("\n=== SUMMARY BY K (mean ± std) ===")
    print(summary_df)


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
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ----------------------------
# 0) CONFIG
# ----------------------------
EXCEL_PATH = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"
MODEL_NAME = "t5-base"     # เปลี่ยนเป็น "google/flan-t5-small" ก็ได้

SEED = 42
TEST_SIZE = 0.2
K_LIST = [1, 3]            # ปรับเป็น [1,3,5,10] ได้ แต่จะใช้เวลามาก

# Excel column positions: A=id, G=text, H=sentiment
ID_COL_POS = 0
TEXT_COL_POS = 6
SENT_COL_POS = 7

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 2      # "positive"/"negative" สั้นมาก
BATCH_TRAIN = 8            # T5 หนักกว่า BERT -> ลดได้ถ้า memory ไม่พอ
BATCH_EVAL  = 8
EPOCHS = 3

# Binary mapping
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "Negative", 1: "Positive"}

# ----------------------------
# 1) Prompt functions (สำคัญสำหรับ T5)
# ----------------------------
def build_prompt(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    # ให้ชัดเจนว่ามีได้แค่สองคำตอบ
    return f"sentiment (negative or positive): {text}"

def normalize_pred_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(ch for ch in s if ch.isalpha())
    if s.startswith("neg"):
        return "negative"
    if s.startswith("pos"):
        return "positive"
    # fallback: ถ้า generate เพี้ยน ให้เป็น negative หรือ positive ก็ได้ (แนะนำให้เป็น negative)
    return "negative"

def preds_to_ids(decoded_texts):
    norm = [normalize_pred_label(x) for x in decoded_texts]
    return np.array([LABEL2ID.get(x, 0) for x in norm], dtype=int)

# ----------------------------
# 2) Load + clean data
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

    df = df.dropna(subset=["id", "text"]).copy()
    df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    df = df[df["text"] != ""].copy()

    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    # keep only pos/neg
    df = df[df["sentiment"].isin(LABEL2ID.keys())].copy()

    df["label"] = df["sentiment"].map(LABEL2ID).astype(int)
    df = df.reset_index(drop=True)
    return df

# ----------------------------
# 3) Build HF dataset (seq2seq)
# ----------------------------
def make_hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    hf = Dataset.from_pandas(df[["id", "text", "sentiment", "label"]].reset_index(drop=True))

    def preprocess(batch):
        inputs = [build_prompt(t) for t in batch["text"]]
        targets = [str(s).strip().lower() for s in batch["sentiment"]]

        # ✅ วิธีที่ไม่ใช้ as_target_tokenizer (แก้ปัญหา T5Tokenizer ไม่มี method)
        # transformers รุ่นใหม่ใช้ text_target ได้
        try:
            enc = tokenizer(
                inputs,
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                padding=False,
                text_target=targets,
                max_target_length=MAX_TARGET_LENGTH,
            )
            return enc
        except TypeError:
            # fallback ถ้า transformers รุ่นเก่ามาก (ไม่รองรับ text_target)
            model_inputs = tokenizer(
                inputs,
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                padding=False,
            )
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    hf = hf.map(preprocess, batched=True)
    return hf

# ----------------------------
# 4) Metrics (decode generated text)
# ----------------------------
def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        # preds: generated token ids (เพราะ predict_with_generate=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # labels: token ids (มี -100)
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
    return compute_metrics

# ----------------------------
# 5) Train one fold
# ----------------------------
def train_one_fold(train_df, val_df, test_df, fold_name: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = make_hf_dataset(train_df, tokenizer)
    val_ds   = make_hf_dataset(val_df, tokenizer)
    test_ds  = make_hf_dataset(test_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    out_dir = os.path.join("t5_kfold_outputs", fold_name)
    os.makedirs(out_dir, exist_ok=True)

    # ✅ สำคัญ: ปิด save checkpoint ระหว่างเทรน ลดปัญหา "No space left on device"
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="no",              # << ปิดการ save ระหว่าง epoch
        learning_rate=3e-4,              # T5 มักใช้ lr สูงกว่า BERT
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        tokenizer=tokenizer,  # ถ้า transformers รุ่นคุณมี warning ก็ไม่เป็นไร (ยังใช้ได้)
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
# 6) Run K-fold (K=1,3,...)
# ----------------------------
def run_kfold(df: pd.DataFrame):
    trainval_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )

    print(f"\nTotal rows   : {len(df)}")
    print(f"TrainVal rows: {len(trainval_df)}")
    print(f"Test rows    : {len(test_df)}")

    all_rows = []

    for K in K_LIST:
        print("\n" + "=" * 60)
        print(f"RUN K = {K}")
        print("=" * 60)

        if K == 1:
            # K=1 baseline split train/val from trainval
            tr_df, va_df = train_test_split(
                trainval_df, test_size=0.2, random_state=SEED, stratify=trainval_df["label"]
            )
            print(f"K=1 | Train: {len(tr_df)} | Val: {len(va_df)} | Test: {len(test_df)}")

            metrics = train_one_fold(tr_df, va_df, test_df, fold_name="K1_single")

            all_rows.append({
                "K": 1, "fold": 1,
                "train_n": len(tr_df), "val_n": len(va_df), "test_n": len(test_df),
                **metrics
            })
            continue

        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

        X = trainval_df["text"].values
        y = trainval_df["label"].values

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            tr_df = trainval_df.iloc[tr_idx].copy()
            va_df = trainval_df.iloc[va_idx].copy()

            print(f"Fold {fold}/{K} | Train: {len(tr_df)} | Val: {len(va_df)} | Test: {len(test_df)}")

            metrics = train_one_fold(tr_df, va_df, test_df, fold_name=f"K{K}_fold{fold}")

            all_rows.append({
                "K": K, "fold": fold,
                "train_n": len(tr_df), "val_n": len(va_df), "test_n": len(test_df),
                **metrics
            })

    results_df = pd.DataFrame(all_rows)

    summary_df = (
        results_df
        .groupby("K")[["eval_accuracy", "eval_f1_macro", "eval_precision_macro", "eval_recall_macro", "train_seconds", "test_seconds"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return results_df, summary_df

# ----------------------------
# 7) MAIN
# ----------------------------
if __name__ == "__main__":
    os.makedirs("t5_kfold_outputs", exist_ok=True)

    df = load_dataset_from_excel(EXCEL_PATH)
    print("Label counts:")
    print(df["sentiment"].value_counts())

    results_df, summary_df = run_kfold(df)

    results_csv = os.path.join("t5_kfold_outputs", "t5_kfold_results_all_folds.csv")
    summary_csv = os.path.join("t5_kfold_outputs", "t5_kfold_summary_by_K.csv")

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(" -", results_csv)
    print(" -", summary_csv)

    print("\n=== SUMMARY BY K (mean ± std) ===")
    print(summary_df)

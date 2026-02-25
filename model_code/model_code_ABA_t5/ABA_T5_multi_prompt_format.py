# ============================================================
# ABA_T5_multitask.py | Propmt formatting for multitask T5 fine-tuning on the ABA dataset.
# Multitask T5 (text-to-text): sentiment + topic + selected
# - Train ONE model on mixed tasks using
# - Then predict + save CSV per task
# - Uses eval_strategy (older transformers) + no save per epoch by default
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd

import evaluate
import torch
import hashlib


from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

# =============================
# CONFIG
# =============================
MODEL_NAME = "t5-base"
DATA_PATH  = "/Users/bbbbben/Desktop/Project in Japan/Task1/ABA Dataset (remove off).xlsx"

SEED = 42
TEST_SIZE = 0.2

DEBUG_N_ROWS = None  # เช่น 500 หรือ None = ทั้งหมด

# save
OUT_DIR = "./mt_outputs"
PRED_DIR = os.path.join(OUT_DIR, "pred_csv")
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
MODEL_DIR = os.path.join(OUT_DIR, "model")
TOK_DIR   = os.path.join(OUT_DIR, "tokenized_cache")
DBG_DIR   = os.path.join(OUT_DIR, "debug_csv")


# tasks
TASKS = ["sentiment", "topic", "selected"]

TOPIC_LABELS = [
    "Booking-issue", "Check-in", "Check-out", "Facility", "Food",
    "Location", "Price", "Room", "Staff", "Taxi-issue"
]

MAX_INPUT_LEN = 256
MAX_TARGET_LEN = {
    "sentiment": 4,
    "topic": 16,
    "selected": 128, 
}

# train baseline
EPOCHS = 3
TRAIN_BS = 8
EVAL_BS  = 8
LR = 3e-4

# generation (ใช้ตอน predict เท่านั้น)
GEN_NUM_BEAMS = 4
GEN_MAX_LEN = {"sentiment": 4, "topic": 16, "selected": 128}

# balancing (ช่วยแก้อาการ "ตอบแต่ positive" / "topic ออกแค่ไม่กี่คลาส")
DO_BALANCE_TASKS = True
BALANCE_STRATEGY = "min"   # "min" = ตัดให้เท่ากับ task ที่น้อยสุด / "cap" = จำกัดเพดาน
CAP_PER_TASK = 2000        # ใช้ถ้า strategy="cap"

# HF download options
LOCAL_FILES_ONLY = False   # ถ้าเน็ตมีปัญหา + เคยโหลดแล้ว เปลี่ยน True ได้

SAVE_STRATEGY = "epoch"    # "no" | "epoch"
SAVE_TOKENIZED_CACHE = True
SAVE_DEBUG_CSV = True

USE_FP16 = torch.cuda.is_available()
USE_BF16 = bool(
    torch.cuda.is_available()
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
)

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOK_DIR, exist_ok=True)
os.makedirs(DBG_DIR, exist_ok=True)

# =============================
# Normalizers
# =============================
def norm_basic(x: str) -> str:
    return str(x).strip().lower()

def norm_sentiment(x: str) -> str:
    x = norm_basic(x)
    if x in ["positive", "pos", "1", "true"]:
        return "positive"
    if x in ["negative", "neg", "0", "false"]:
        return "negative"
    if "positive" in x and "negative" not in x:
        return "positive"
    if "negative" in x and "positive" not in x:
        return "negative"
    return x

def norm_topic(x: str) -> str:
    x = str(x).strip()
    mapping = {t.lower(): t for t in TOPIC_LABELS}
    return mapping.get(x.lower(), x)

def force_sentiment(pred: str) -> str:
    p = norm_sentiment(pred)
    return p if p in ["positive", "negative"] else "unknown"

def force_topic(pred: str) -> str:
    p = norm_topic(pred)
    return p if p in TOPIC_LABELS else "unknown"

# =============================
# Prompt builder (MULTITASK)
# =============================
def build_input(text: str, task: str, topic: str | None = None, sentiment: str | None = None, selected: str | None = None) -> str:
    text = str(text).replace("\n", " ").strip()

    if task == "sentiment":
        t = topic if topic else "unknown"
        return (
            "Task: Sentiment classification.\n"
            "Given a review and a specific topic, "
            "predict the sentiment towards that topic.\n"
            "Allowed labels: positive, negative\n"
            "Rules: Answer with exactly ONE label from the allowed list. No extra words.\n"
            f"Topic: {t}\n"
            f"Review: {text}\n"
            "Answer:"
        )

    if task == "topic":
        sp = selected if selected else "unknown"
        allowed = ", ".join(TOPIC_LABELS)
        return (
            "Task: Topic classification.\n"
            "Given a review and a supporting phrase, "
            "predict the topic of that phrase.\n"
            f"Allowed topics: {allowed}\n"
            "Rules: Answer with exactly ONE topic from the list. No extra words.\n"
            f"Supporting phrase: {sp}\n"
            f"Review: {text}\n"
            "Answer:"
        )

    if task == "selected":
        t = topic if topic else "unknown"
        s = sentiment if sentiment else "unknown"
        return (
            "Task: Extract a short supporting phrase for the review.\n"
            "Rules:\n"
            "- ONLY use words that appear in the review (you can drop words, but don't add new ones).\n"
            "- You may reorder slightly, but try to keep the original wording.\n"
            "- Output 1–2 sentences.\n"
            f"Topic: {t}\n"
            f"Sentiment: {s}\n"
            f"Review: {text}\n"
            "Supporting phrase:"
        )

    return text

# =============================
# Load Excel (same mapping as before)
# A=0 (ID), E=4 (text), F=5 (topic), G=6 (selected), H=7 (sentiment)
# =============================
df = pd.read_excel(DATA_PATH)
if DEBUG_N_ROWS is not None:
    df = df.head(int(DEBUG_N_ROWS)).copy()

ID_COL       = df.columns[0]
TEXT_COL     = df.columns[4]
TOPIC_COL    = df.columns[5]
SELECTED_COL = df.columns[6]
SENT_COL     = df.columns[7]

# =============================
# Build per-task dataframe (keep needed fields)
# =============================
def make_task_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "sentiment":
        # use (text, topic) -> sentiment
        sub = df[[ID_COL, TEXT_COL, TOPIC_COL, SENT_COL]].dropna().copy()
        sub.rename(columns={
            ID_COL: "id",
            TEXT_COL: "text",
            TOPIC_COL: "topic",
            SENT_COL: "sentiment",
        }, inplace=True)
        sub["topic"] = sub["topic"].apply(norm_topic)
        sub["sentiment"] = sub["sentiment"].apply(norm_sentiment)

        sub = sub[sub["topic"].isin(TOPIC_LABELS)].copy()
        sub = sub[sub["sentiment"].isin(["positive", "negative"])].copy()

        sub["task"] = "sentiment"
        sub["selected"] = ""            # placeholder
        sub["target"] = sub["sentiment"]

        # key must include topic (because same review has multiple topics)
        sub["key"] = (
            sub["task"].astype(str) + "||" +
            sub["id"].astype(str) + "||" +
            sub["text"].astype(str) + "||" +
            sub["topic"].astype(str)
        )

    elif task == "topic":
        # use (text, selected) -> topic
        sub = df[[ID_COL, TEXT_COL, SELECTED_COL, TOPIC_COL]].dropna().copy()
        sub.rename(columns={
            ID_COL: "id",
            TEXT_COL: "text",
            SELECTED_COL: "selected",
            TOPIC_COL: "topic",
        }, inplace=True)
        sub["topic"] = sub["topic"].apply(norm_topic)
        sub["selected"] = sub["selected"].astype(str)

        sub = sub[sub["topic"].isin(TOPIC_LABELS)].copy()

        sub["task"] = "topic"
        sub["sentiment"] = ""          # placeholder
        sub["target"] = sub["topic"]

        # key include selected to avoid collisions
        sub["key"] = (
            sub["task"].astype(str) + "||" +
            sub["id"].astype(str) + "||" +
            sub["text"].astype(str) + "||" +
            sub["selected"].astype(str)
        )

    else:  # selected
        sub = df[[ID_COL, TEXT_COL, SELECTED_COL, TOPIC_COL, SENT_COL]].dropna().copy()
        sub.rename(columns={
            ID_COL: "id",
            TEXT_COL: "text",
            SELECTED_COL: "selected",
            TOPIC_COL: "topic",
            SENT_COL: "sentiment"
        }, inplace=True)

        sub["topic"] = sub["topic"].apply(norm_topic)
        sub["sentiment"] = sub["sentiment"].apply(norm_sentiment)
        sub["selected"] = sub["selected"].astype(str)

        sub = sub[sub["topic"].isin(TOPIC_LABELS)].copy()
        sub = sub[sub["sentiment"].isin(["positive", "negative"])].copy()

        sub["task"] = "selected"
        sub["target"] = sub["selected"]

        # key include topic+sentiment
        sub["key"] = (
            sub["task"].astype(str) + "||" +
            sub["id"].astype(str) + "||" +
            sub["text"].astype(str) + "||" +
            sub["topic"].astype(str) + "||" +
            sub["sentiment"].astype(str)
        )

    # unify columns
    if "topic" not in sub.columns:
        sub["topic"] = ""
    if "sentiment" not in sub.columns:
        sub["sentiment"] = ""
    if "selected" not in sub.columns:
        sub["selected"] = ""

    sub = sub.reset_index(drop=True)
    return sub[["task","id","text","topic","sentiment","selected","target","key"]].copy()

task_dfs = {t: make_task_df(df, t) for t in TASKS}
for t in TASKS:
    print(f"{t}: {len(task_dfs[t])} rows")

# =============================
# (Optional) Balance tasks to reduce collapse
# =============================
# ถ้าไม่ balance อาจเจออาการ "ตอบแต่ positive" หรือ "topic ออกแค่ไม่กี่คลาส" เพราะบาง task มีข้อมูลเยอะกว่า
def balance_task_dfs(task_dfs: dict) -> pd.DataFrame:
    if not DO_BALANCE_TASKS:
        return pd.concat(list(task_dfs.values()), ignore_index=True)

    sizes = {k: len(v) for k, v in task_dfs.items()}

    if BALANCE_STRATEGY == "min":
        n = min(sizes.values())
        out = []
        for k, dfk in task_dfs.items():
            out.append(dfk.sample(n=n, random_state=SEED) if len(dfk) > n else dfk)
        return pd.concat(out, ignore_index=True)

    if BALANCE_STRATEGY == "cap":
        out = []
        for k, dfk in task_dfs.items():
            n = min(len(dfk), CAP_PER_TASK)
            out.append(dfk.sample(n=n, random_state=SEED) if len(dfk) > n else dfk)
        return pd.concat(out, ignore_index=True)

    return pd.concat(list(task_dfs.values()), ignore_index=True)

mt_df = balance_task_dfs(task_dfs).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"\nMULTITASK mixed rows: {len(mt_df)} (balanced={DO_BALANCE_TASKS}, strategy={BALANCE_STRATEGY})")

# =============================
# Deterministic split (safe)
# - regenerate if saved keys don't match current dataset
# =============================
def fingerprint_keys(keys: list[str]) -> str:
    # cheap stable fingerprint
    h = hashlib.md5()
    sample = (keys[:200] if len(keys) >= 200 else keys)
    h.update(("|".join(sample)).encode("utf-8", errors="ignore"))
    return f"n{len(keys)}_{h.hexdigest()[:8]}"

FP = fingerprint_keys(mt_df["key"].tolist())
SPLIT_PATH = os.path.join(SPLIT_DIR, f"split_multitask_{FP}_seed{SEED}_test{str(TEST_SIZE).replace('.','p')}.json")

def get_split_keys(keys: list[str]) -> tuple[set[str], set[str]]:
    keyset = set(keys)
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tr = set(obj.get("train_keys", []))
        ev = set(obj.get("eval_keys", []))
        # if mismatch -> regenerate
        if tr.issubset(keyset) and ev.issubset(keyset) and len(tr) and len(ev):
            return tr, ev

    n = len(keys)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_eval = int(round(n * TEST_SIZE))

    eval_idx = set(idx[:n_eval].tolist())
    train_idx = set(idx[n_eval:].tolist())

    train_keys = [keys[i] for i in train_idx]
    eval_keys  = [keys[i] for i in eval_idx]

    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump({"train_keys": train_keys, "eval_keys": eval_keys}, f, ensure_ascii=False, indent=2)

    return set(train_keys), set(eval_keys)

train_keys, eval_keys = get_split_keys(mt_df["key"].tolist())
train_df = mt_df[mt_df["key"].isin(train_keys)].copy()
eval_df  = mt_df[mt_df["key"].isin(eval_keys)].copy()
print(f"Train rows: {len(train_df)} | Eval rows: {len(eval_df)}")

# =============================
# HF Dataset
# =============================
def df_to_dataset(d: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(d[["task","id","text","topic","sentiment","selected","target","key"]], preserve_index=False)

train_ds = df_to_dataset(train_df)
eval_ds  = df_to_dataset(eval_df)

# =============================
# Tokenizer / model
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)

# =============================
# Preprocess (build input by task)
# =============================
def preprocess(example):
    task = example["task"]

    inp = build_input(
        text=example["text"],
        task=task,
        topic=example.get("topic", None),
        sentiment=example.get("sentiment", None),
        selected=example.get("selected", None),
    )

    tgt = example["target"]
    if task == "sentiment":
        tgt = norm_sentiment(tgt)
    elif task == "topic":
        tgt = norm_topic(tgt)

    model_inputs = tokenizer(inp, truncation=True, max_length=MAX_INPUT_LEN)
    labels = tokenizer(text_target=str(tgt), truncation=True, max_length=MAX_TARGET_LEN[task])
    model_inputs["labels"] = labels["input_ids"]

    model_inputs["debug_input"] = inp
    model_inputs["debug_target"] = str(tgt)
    return model_inputs

train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

if SAVE_TOKENIZED_CACHE:
    train_tok.save_to_disk(os.path.join(TOK_DIR, "train_tok"))
    eval_tok.save_to_disk(os.path.join(TOK_DIR, "eval_tok"))
    print(f"✅ Saved tokenized datasets to: {TOK_DIR}")

def _sorted_by_id(df_: pd.DataFrame) -> pd.DataFrame:
    # พยายามเรียงแบบตัวเลข ถ้า id เป็น numeric; ถ้าไม่ได้ก็เรียงแบบ string
    tmp = df_.copy()
    tmp["_id_num"] = pd.to_numeric(tmp["id"], errors="coerce")
    if tmp["_id_num"].notna().any():
        tmp = tmp.sort_values(["_id_num", "id"], ascending=True)
    else:
        tmp = tmp.sort_values(["id"], ascending=True)
    return tmp.drop(columns=["_id_num"], errors="ignore").reset_index(drop=True)

def make_debug_df_from_raw(raw_df: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = raw_df[raw_df["task"] == task].copy()
    sub = _sorted_by_id(sub)

    # สร้าง debug_input / debug_target ให้เหมือน preprocess
    debug_inputs = []
    debug_targets = []
    for _, r in sub.iterrows():
        inp = build_input(
            text=r["text"],
            task=task,
            topic=r.get("topic", None),
            sentiment=r.get("sentiment", None),
            selected=r.get("selected", None),
        )
        tgt = str(r["target"])
        if task == "sentiment":
            tgt = norm_sentiment(tgt)
        elif task == "topic":
            tgt = norm_topic(tgt)

        debug_inputs.append(inp)
        debug_targets.append(tgt)

    out = sub[["task","id","topic","sentiment","selected"]].copy()
    out["debug_input"] = debug_inputs
    out["debug_target"] = debug_targets
    return out

if SAVE_DEBUG_CSV:
    for split_name, raw_df in [("train", train_df), ("eval", eval_df)]:
        for task in TASKS:
            dbg = make_debug_df_from_raw(raw_df, task)
            out_path = os.path.join(DBG_DIR, f"debug_{split_name}_{task}_inputs.csv")
            dbg.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved debug CSV: {out_path}")

    print(f"✅ Saved debug CSV to: {DBG_DIR}")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# =============================
# Train
# =============================
args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy=SAVE_STRATEGY,
    report_to="none",
    logging_strategy="steps",
    logging_steps=100,

    predict_with_generate=True,
    generation_max_length=max(MAX_TARGET_LEN.values()),

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=LR,

    fp16=USE_FP16,
    bf16=USE_BF16,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
    compute_metrics=lambda _: {},
)

trainer.train()

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Saved final model to: {MODEL_DIR}")

# =============================
# Safe pred ids
# =============================
def safe_get_pred_ids(preds):
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.asarray(preds)
    if preds.ndim == 3:
        preds = preds.argmax(axis=-1)
    preds = preds.astype(np.int64)
    preds[preds < 0] = 0
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        preds[preds >= vocab_size] = vocab_size - 1
    return preds

rouge = evaluate.load("rouge")

# =============================
# Predict per task (full, unbalanced)
# =============================
def predict_one_task(task: str):
    full_df = task_dfs[task].copy()
    full_ds = df_to_dataset(full_df)

    def preprocess_task(example):
        inp = build_input(
            text=example["text"],
            task=task,
            topic=example.get("topic", None),
            sentiment=example.get("sentiment", None),
            selected=example.get("selected", None),
        )
        tgt = example["target"]
        if task == "sentiment":
            tgt = norm_sentiment(tgt)
        elif task == "topic":
            tgt = norm_topic(tgt)

        mi = tokenizer(inp, truncation=True, max_length=MAX_INPUT_LEN)
        lab = tokenizer(text_target=str(tgt), truncation=True, max_length=MAX_TARGET_LEN[task])
        mi["labels"] = lab["input_ids"]
        mi["debug_input"] = inp
        mi["debug_target"] = str(tgt)
        return mi

    full_tok = full_ds.map(preprocess_task, remove_columns=full_ds.column_names)

    pred_out = trainer.predict(
        full_tok,
        max_length=GEN_MAX_LEN[task],
        num_beams=GEN_NUM_BEAMS,
    )

    pred_ids = safe_get_pred_ids(pred_out.predictions)
    raw_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    trues = full_df["target"].astype(str).tolist()

    if task == "sentiment":
        trues = [norm_sentiment(x) for x in trues]
        forced = [force_sentiment(x) for x in raw_preds]
        match = [(t == p) for t, p in zip(trues, forced)]
        metrics = {"accuracy": float(np.mean(match)), "unknown_rate": float(np.mean([p == "unknown" for p in forced]))}
    elif task == "topic":
        trues = [norm_topic(x) for x in trues]
        forced = [force_topic(x) for x in raw_preds]
        match = [(t == p) for t, p in zip(trues, forced)]
        metrics = {"accuracy": float(np.mean(match)), "unknown_rate": float(np.mean([p == "unknown" for p in forced]))}
    else:
        forced = raw_preds
        metrics = rouge.compute(predictions=forced, references=trues)

    out = full_df.copy()
    out["raw_pred"] = raw_preds
    out["forced_pred"] = forced

    if task in ["sentiment", "topic"]:
        out["match"] = (out["forced_pred"].astype(str) == pd.Series(trues).astype(str)).astype(int)
    else:
        out["exact_match"] = (out["raw_pred"].astype(str) == pd.Series(trues).astype(str)).astype(int)

    csv_path = os.path.join(PRED_DIR, f"full_preds_{task}_multitask.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved predictions: {csv_path}")
    print(f"📌 {task} metrics:", metrics)
    return {"task": task, **metrics}

summary = [predict_one_task(t) for t in TASKS]
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(PRED_DIR, "SUMMARY_multitask.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"\n✅ Saved summary: {summary_path}")
print(summary_df)
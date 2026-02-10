- Auto fine-tune มั้ย

pipeline: 

(raw text)
   ↓
สร้าง multiple datasets (sentiment (label)/ topic (label)/ selected content (text))
   ↓
สร้าง 2 input styles
   - prefix
   - prompt-styled
   ↓
T5 fine-tune (HF Trainer)
   ↓
(optional) auto hyperparameter
   ↓
compare performance

#Prefix
sentiment classification: {text}
topic classification: {text}
generate selected contentn: [text]

#Prompt-styled
Task: Determine the sentiment of the following review.
Review: {text}
Sentiment:

Task: Identify the topic of the following review.
Review: {text}
Topic:

Task: Generate the selected content from the following review.
Review: {text}
Answer:

#กำหนด prompt
def build_input(text, task, style):
    if style == "prefix":
        if task == "sentiment":
            return f"sentiment classification: {text}"
        elif task == "topic":
            return f"topic classification: {text}"
        elif task == "selected":
            return f"generate selected content: {text}"

    elif style == "prompt":
        if task == "sentiment":
            return (
                "Task: Determine the sentiment of the following review.\n"
                f"Review: {text}\nSentiment:"
            )
        elif task == "topic":
            return (
                "Task: Identify the topic of the following review.\n"
                f"Review: {text}\nTopic:"
            )
        elif task == "selected":
            return (
                "Task: Generate the selected content that supports the opinion.\n"
                f"Review: {text}\nAnswer:"
            )


#code คับ

# ==============================
# ABA-T5 + HF Trainer + Optuna
# ==============================

import pandas as pd
import numpy as np
import evaluate
import optuna

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    set_seed
)

# ------------------------------
# CONFIG (ปรับตรงนี้พอ)
# ------------------------------
MODEL_NAME = "t5-base"
DATA_PATH = ""      # <- ไฟล์ .xlsx
TEXT_COL = "text"            # column raw text

TASK = "sentiment"           # sentiment | topic | selected
STYLE = "prompt"             # prefix | prompt

SEED = 42
N_TRIALS = 5                 # auto fine-tune trials

# column label ตาม task
TASK_CONFIG = {
    "sentiment": "sentiment",
    "topic": "topic",
    "selected": "selected_text",
}

# ------------------------------
# Seed
# ------------------------------
set_seed(SEED)

# ------------------------------
# Prompt / Prefix
# ------------------------------
def build_input(text, task, style):
    if style == "prefix":
        if task == "sentiment":
            return f"sentiment classification: {text}"
        elif task == "topic":
            return f"topic classification: {text}"
        elif task == "selected":
            return f"generate selected content: {text}"

    if style == "prompt":
        if task == "sentiment":
            return (
                "Task: Determine the sentiment of the following review.\n"
                f"Review: {text}\nSentiment:"
            )
        elif task == "topic":
            return (
                "Task: Identify the topic of the following review.\n"
                f"Review: {text}\nTopic:"
            )
        elif task == "selected":
            return (
                "Task: Generate the selected content that supports the opinion.\n"
                f"Review: {text}\nAnswer:"
            )

# ------------------------------
# Load Dataset (.xlsx)
# ------------------------------
df = pd.read_excel(DATA_PATH).dropna()
dataset = Dataset.from_pandas(df)

# ------------------------------
# Tokenizer / Model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ------------------------------
# Preprocess
# ------------------------------
def preprocess(example):
    text = example[TEXT_COL]
    label = str(example[TASK_CONFIG[TASK]])

    model_input = build_input(text, TASK, STYLE)

    inputs = tokenizer(
        model_input,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=64
        )

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

# ------------------------------
# Metric (classification only)
# ------------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return accuracy.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

# ------------------------------
# TrainingArguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none",
)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics if TASK != "selected" else None,
)

# ------------------------------
# Optuna search space
# ------------------------------
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16]
        ),
    }

# ------------------------------
# Auto fine-tune
# ------------------------------
best_run = trainer.hyperparameter_search(
    direction="maximize" if TASK != "selected" else "minimize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=N_TRIALS,
)

print("✅ Best hyperparameters")
print(best_run.hyperparameters)
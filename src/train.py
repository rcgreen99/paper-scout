import logging

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch
import numpy as np


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_dataset(dataset_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """Loads, tokenizes, and returns the full dataset from a JSON file."""
    logging.info("Loading dataset from %s", dataset_path)
    # Load the dataset
    ds = load_dataset("json", data_files=dataset_path, split="train")

    # Keep only abstracts
    ds = ds.remove_columns([c for c in ds.column_names if c != "abstract"])
    ds = ds.filter(lambda ex: ex["abstract"] is not None and len(ex["abstract"]) > 50)

    # Tokenize abstracts
    tokenizer.model_max_length = 64  # Can try 256 later

    def tok(examples):
        return tokenizer(
            examples["abstract"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )

    tokenized = ds.map(
        tok,
        batched=True,
        remove_columns=["abstract"],
        desc="Tokenizing abstracts",
    )
    tokenized.set_format("torch")
    return tokenized


def get_model(model_name: str) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    logging.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute token‐level MLM accuracy, masking out all non‐MLM tokens (where label == -100).
    """
    preds, labels = eval_pred.predictions, eval_pred.label_ids

    # If preds came in as a tuple (some models do that), unpack it
    if isinstance(preds, tuple):
        preds = preds[0]

    # Mask out all positions where labels == -100 (i.e. non-MLM tokens)
    mask = labels != -100
    masked_preds = preds[mask]
    masked_labels = labels[mask]

    # Compute mean accuracy
    acc = (masked_preds == masked_labels).mean()
    return {"accuracy": float(acc)}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def train(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
):
    logging.info("Defining LoRA config...")
    # 1) Define LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 2) Wrap model
    model = get_peft_model(model, lora_config).to("cuda")
    model.print_trainable_parameters()

    # 3) Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="lora_arxiv_mlm",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        num_train_epochs=3,
        fp16=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=10000,
        # logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Saving final model...")
    trainer.save_model("lora_arxiv_mlm_final")


if __name__ == "__main__":
    DATASET_PATH = "data/arxiv-metadata-oai-snapshot.json"
    # MODEL_NAME = "allenai/scibert_scivocab_uncased"
    MODEL_NAME = "google-bert/bert-base-uncased"

    # 1) Load model & tokenizer
    model, tokenizer = get_model(MODEL_NAME)

    # 2) Prepare tokenized dataset
    tokenized_ds = get_dataset(DATASET_PATH, tokenizer)

    # 3) Split into train/validation
    # ds_dict = tokenized_ds.train_test_split(test_size=0.1, seed=42)
    debug_ds = tokenized_ds.shuffle(seed=42).select(range(1_000_000))
    # debug_ds = tokenized_ds.shuffle(seed=42).select(range(10_000))
    # debug_ds = tokenized_ds.shuffle(seed=42).select(range(1_000))
    # debug_ds = tokenized_ds.shuffle(seed=42).select(range(100))
    ds_dict = debug_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds_dict["train"]
    eval_ds = ds_dict["test"]
    # 4) Train with validation
    train(model, tokenizer, train_ds, eval_ds)

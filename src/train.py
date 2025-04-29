import logging

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_dataset(dataset_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """Loads, tokenizes, and tokenizes the dataset from a JSON file."""
    logging.info("Loading dataset...")
    # Load the dataset
    ds = load_dataset("json", data_files=dataset_path, split="train")

    # Keep only abstracts (drop everything else)
    ds = ds.remove_columns([c for c in ds.column_names if c != "abstract"])
    ds = ds.filter(lambda ex: ex["abstract"] is not None and len(ex["abstract"]) > 50)

    def tok(examples):
        return tokenizer(examples["abstract"], truncation=True, max_length=256)

    # Tokenize
    tokenizer.model_max_length = 256
    tokenized = ds.map(
        tok,
        batched=True,
        remove_columns=["abstract"],
        desc="Running tokenizer on dataset",
    )
    tokenized.set_format("torch")
    return tokenized


def get_model(model_name: str) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    logging.info("Loading model %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def train(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
):
    logging.info("Defining LoRA config...")
    # 1) Define your LoRA config
    lora_config = LoraConfig(
        r=8,  # bottleneck rank
        lora_alpha=32,  # scaling
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
    )

    # 2) Wrap your model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Split dataset??
    # ds_dict = dataset.train_test_split(test_size=0.1)
    # train_examples = ds_dict["train"]
    # val_examples = ds_dict["test"]

    # 5) data collator for masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=0.15
    )

    # 6) training arguments
    training_args = TrainingArguments(
        output_dir="lora_arxiv_mlm",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=200,
        save_total_limit=2,
        save_steps=1000,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logging.info("Training...")
    trainer.train()
    logging.info("Saving model...")
    trainer.save_model("lora_arxiv_mlm_final")


if __name__ == "__main__":
    DATASET_PATH = "data/arxiv-metadata-oai-snapshot.json"
    MODEL_NAME = "allenai/scibert_scivocab_uncased"

    model, tokenizer = get_model(MODEL_NAME)
    tokenized_ds = get_dataset(DATASET_PATH, tokenizer)
    train(model, tokenizer, tokenized_ds)

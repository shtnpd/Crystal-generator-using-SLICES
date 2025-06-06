from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import pandas as pd

slices_txt   = "data/processed/omat.txt"

dataset = load_dataset("text", data_files=str(slices_txt))["train"]

split = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

print("Train size:", len(train_ds))
print("Validation size:", len(val_ds))

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_slices(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

train_tok = train_ds.map(tokenize_slices, batched=True, remove_columns=["text"])
val_tok   = val_ds.map(tokenize_slices, batched=True, remove_columns=["text"])

train_tok = train_tok.rename_column("input_ids", "labels")
val_tok   = val_tok.rename_column("input_ids", "labels")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

def data_collator(batch):
    input_ids = torch.stack([torch.tensor(ex["labels"]) for ex in batch])
    attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in batch])
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         input_ids.clone(),
    }
# data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

training_args = TrainingArguments(
    output_dir="checkpoints/pretrain_gpt2_slices_2",
    num_train_epochs=4,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=4,

    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps= 2000,

    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=10,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    prediction_loss_only=False,

    fp16=True,
    # gradient_checkpointing=True,

    dataloader_num_workers=4,
    dataloader_pin_memory=True,

    ddp_find_unused_parameters=False,

    logging_dir="logs/pretrain_2",

)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    callbacks=[early_stopping],
)

trainer.train()

# print(trainer.state.global_step)
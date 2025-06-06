from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

PROJECT_ROOT = Path.cwd()
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

# TRAIN_CSV = DATA_DIR / "MP20_train_with_slices.csv"
# VAL_CSV   = DATA_DIR / "MP20_val_with_slices.csv"
TRAIN_CSV = DATA_DIR / "MP20-train-slices-trimmed.csv"
VAL_CSV   = DATA_DIR / "MP20-val-slices-trimmed.csv"

assert TRAIN_CSV.exists(), f"Файл {TRAIN_CSV} не найден"
assert VAL_CSV.exists(),   f"Файл {VAL_CSV} не найден"

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

for col in ["band_gap", "formation_energy", "slices_string"]:
    assert col in train_df.columns, f"В train_df отсутствует колонка {col}"
    assert col in val_df.columns,   f"В val_df отсутствует колонка {col}"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

bg_tokens = [f"<BG_{i/10:.1f}>" for i in range(0, 101)]
fe_tokens = [f"<FE_{(i-50)/10:+.1f}>" for i in range(0, 101)]

basic_specials = ["<BOS>", "<EOS>", "|"]

all_specials = basic_specials + bg_tokens + fe_tokens

tokenizer.add_special_tokens({"additional_special_tokens": all_specials})
tokenizer.pad_token = tokenizer.eos_token

class SlicesConditionalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: GPT2TokenizerFast, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bg_val = float(row["band_gap"])
        fe_val = float(row["formation_energy"])

        i_bg = round(bg_val / 0.1)
        i_fe = round((fe_val + 5.0) / 0.1)

        bg_tok = f"<BG_{i_bg/10:.1f}>"
        fe_tok = f"<FE_{(i_fe-50)/10:+.1f}>"

        slices_txt = row["slices_string"].strip()
        text = f"<BOS> {bg_tok} {fe_tok} | {slices_txt} <EOS>"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)       
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


def fine_tune(pretrain_dir_name, output_dir_name, logging_dir_name, num_train_epochs=6, learning_rate=3e-5, 
warmup_steps=400, logging_steps=100, save_steps=200, eval_steps=200, save_total_limit=3, weight_decay=0.01):
    train_dataset = SlicesConditionalDataset(train_df, tokenizer, max_length=512)
    val_dataset   = SlicesConditionalDataset(val_df,   tokenizer, max_length=512)

    PRETRAINED_CKPT = f"checkpoints/{pretrain_dir_name}"
    model = GPT2LMHeadModel.from_pretrained(PRETRAINED_CKPT)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{output_dir_name}",
        overwrite_output_dir=True,

        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=4,

        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,

        save_steps=save_steps,
        save_total_limit=save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # prediction_loss_only=False,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        ddp_find_unused_parameters=False,

        report_to=["tensorboard"],

        fp16=True,
        logging_dir=f"logs/{logging_dir_name}",
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=None,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is None:
        best_ckpt = training_args.output_dir

    tokenizer.save_pretrained(best_ckpt)
    model.save_pretrained(best_ckpt)
    print("Fine-tuning is done. Best checkpoint:", best_ckpt)

# fine_tune(pretrain_dir_name="pretrain_gpt2_slices_2/checkpoint-5000",
#           output_dir_name="finetune_gpt2_slices_4",
#           logging_dir_name="finetune_4",
#           num_train_epochs=8,
#           learning_rate=3e-4,
#           warmup_steps=1500,
#           logging_steps=100,
#           save_steps=200,
#           eval_steps=200,
#           save_total_limit=3)

fine_tune(pretrain_dir_name="pretrain_gpt2_slices_2/checkpoint-5000",
          output_dir_name="finetune_gpt2_slices_5",
          logging_dir_name="finetune_5",
          num_train_epochs=10,
        #   learning_rate=5e-4,
          learning_rate=1e-3,
        #   warmup_steps=200,
          warmup_steps=0,
          logging_steps=50,
          save_steps=200,
          eval_steps=200,
          save_total_limit=3,
          weight_decay=0.005)
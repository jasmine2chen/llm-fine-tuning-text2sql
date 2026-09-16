"""Fine-tune an LLM for text-to-SQL using Unsloth QLoRA and TRL's SFTTrainer."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer
import yaml
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_wandb(config: dict) -> None:
    """Initialise Weights & Biases if enabled in the config."""
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        import os

        os.environ["WANDB_MODE"] = "disabled"
        return

    import wandb

    wandb.init(project=wandb_cfg.get("project", "text2sql-finetune"))


def train(config_path: str | Path) -> None:
    """Run a full fine-tuning job from a YAML config file.

    Steps
    -----
    1. Load the model in 4-bit via Unsloth's ``FastLanguageModel``.
    2. Attach a LoRA adapter (also through Unsloth).
    3. Prepare the dataset (expects pre-formatted chat JSON).
    4. Train with TRL's ``SFTTrainer``.
    5. Save the adapter weights.
    """
    from unsloth import FastLanguageModel

    config = _load_config(config_path)
    _setup_wandb(config)

    model_id: str = config["model_id"]
    output_dir: str = config["output_dir"]
    max_seq_length: int = config.get("max_seq_length", 2048)
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    # --- model ---------------------------------------------------------
    logger.info("Loading model %s (4-bit quantisation via Unsloth)", model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 128),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # --- data ----------------------------------------------------------
    from text2sql.data import load_and_prepare

    splits = load_and_prepare(config_path)
    train_dataset = splits["train"]

    # --- training args -------------------------------------------------
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        bf16=train_cfg.get("bf16", True),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        max_seq_length=max_seq_length,
        packing=train_cfg.get("packing", True),
        report_to="wandb" if config.get("wandb", {}).get("enabled") else "none",
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 128),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- trainer -------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training (%d epochs)", train_cfg.get("num_epochs", 3))
    trainer.train()
    trainer.save_model()
    logger.info("Adapter saved to %s", output_dir)

    # free memory
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.command()
def main(config: str = typer.Option("configs/train.yaml", "--config", "-c")) -> None:
    """CLI entry-point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    train(config)


if __name__ == "__main__":
    app()

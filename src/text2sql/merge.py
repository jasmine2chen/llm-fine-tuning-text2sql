"""Merge a LoRA adapter into the base model and save the full merged weights."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


def merge_adapter(adapter_path: str | Path, output_path: str | Path | None = None) -> None:
    """Load a PEFT adapter, merge it into the base model, and save.

    Parameters
    ----------
    adapter_path:
        Directory containing the adapter weights (``adapter_config.json``,
        ``adapter_model.safetensors``, etc.).
    output_path:
        Where to write the merged model.  Defaults to *adapter_path* itself
        (in-place merge).
    """
    from peft import AutoPeftModelForCausalLM

    adapter_path = Path(adapter_path)
    if output_path is None:
        output_path = adapter_path
    output_path = Path(output_path)

    logger.info("Loading PEFT model from %s", adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    logger.info("Merging adapter into base model")
    merged = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path), safe_serialization=True, max_shard_size="2GB")
    logger.info("Merged model saved to %s", output_path)

    # Also copy the tokenizer so the merged directory is self-contained
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Tokenizer saved to %s", output_path)


@app.command()
def main(
    adapter_path: str = typer.Argument(..., help="Path to the adapter checkpoint"),
    output_path: str = typer.Option(None, "--output", "-o", help="Output directory for merged model"),
) -> None:
    """CLI entry-point for merging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    merge_adapter(adapter_path, output_path)


if __name__ == "__main__":
    app()

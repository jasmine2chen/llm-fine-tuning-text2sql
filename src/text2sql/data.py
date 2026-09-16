"""Dataset loading and chat message formatting for text-to-SQL fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a text-to-SQL query translator. Users will ask you questions in English "
    "and you will generate a SQL query based on the provided SCHEMA.\n"
    "SCHEMA:\n{schema}"
)


def create_chat_messages(sample: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert a raw dataset sample into chat-format messages.

    Each sample from b-mc2/sql-create-context has:
      - ``question``: natural language question
      - ``context``: CREATE TABLE schema definition(s)
      - ``answer``: gold SQL query

    Returns a dict with a single key ``"messages"`` containing the
    system / user / assistant message list expected by TRL's SFTTrainer.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(schema=sample["context"])},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }


def load_and_prepare(config_path: str | Path) -> DatasetDict:
    """Load the dataset, format into chat messages, and split into train/test.

    Parameters
    ----------
    config_path:
        Path to a YAML config file that must contain at minimum:
        ``dataset_id``, ``dataset_samples``, and ``test_ratio``.

    Returns
    -------
    DatasetDict with ``"train"`` and ``"test"`` splits.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_id: str = config["dataset_id"]
    n_samples: int = config["dataset_samples"]
    test_ratio: float = config["test_ratio"]

    logger.info("Loading dataset %s (sampling %d examples)", dataset_id, n_samples)
    dataset = load_dataset(dataset_id, split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    logger.info("Formatting samples into chat messages")
    dataset = dataset.map(create_chat_messages, remove_columns=dataset.column_names, batched=False)

    splits = dataset.train_test_split(test_size=test_ratio, seed=42)
    logger.info(
        "Split sizes  train=%d  test=%d", len(splits["train"]), len(splits["test"])
    )
    return splits


def save_splits(splits: DatasetDict, output_dir: str | Path) -> None:
    """Persist train/test splits as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "test"):
        path = output_dir / f"{name}_dataset.json"
        splits[name].to_json(str(path), orient="records")
        logger.info("Saved %s split to %s", name, path)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml"
    ds = load_and_prepare(cfg)
    save_splits(ds, ".")

"""W&B hyperparameter sweep for text-to-SQL fine-tuning.

Varies LoRA rank, learning rate, and number of epochs.  Run with::

    python scripts/sweep.py            # creates the sweep and starts an agent
    python scripts/sweep.py --count 9  # run up to 9 trials
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import typer
import wandb
import yaml

from text2sql.train import train

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "train/loss", "goal": "minimize"},
    "parameters": {
        "lora_r": {"values": [64, 128, 256]},
        "learning_rate": {"values": [1e-4, 2e-4, 5e-4]},
        "num_epochs": {"values": [1, 2, 3]},
    },
}


def _run_sweep_trial(base_config_path: str = "configs/train.yaml") -> None:
    """Called by ``wandb.agent`` for each trial."""
    run = wandb.init()
    sweep_params = dict(wandb.config)

    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Override config with sweep parameters
    config["lora"]["r"] = sweep_params["lora_r"]
    config["training"]["learning_rate"] = sweep_params["learning_rate"]
    config["training"]["num_epochs"] = sweep_params["num_epochs"]

    # Write a temporary config for this trial
    tmp_config = Path(f"/tmp/sweep_config_{run.id}.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(config, f)

    try:
        train(str(tmp_config))
    finally:
        tmp_config.unlink(missing_ok=True)
        run.finish()


@app.command()
def main(
    project: str = typer.Option("text2sql-finetune", "--project", "-p"),
    count: int = typer.Option(9, "--count", "-n", help="Max number of sweep trials"),
) -> None:
    """Create a W&B sweep and launch an agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    logger.info("Created sweep %s in project %s", sweep_id, project)
    wandb.agent(sweep_id, function=_run_sweep_trial, count=count)


if __name__ == "__main__":
    app()

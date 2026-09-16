"""Execution-based SQL evaluation for text-to-SQL models.

The key insight: exact string match is fundamentally broken for evaluating
SQL generation.  ``SELECT a, b FROM t`` and ``SELECT a,b FROM t`` are identical
queries, and even structurally different queries can return the same result set.

This module provides three complementary metrics:

* **exact_match** -- normalised string comparison (baseline).
* **execution_accuracy** -- both predicted and reference SQL are executed
  against an in-memory SQLite database built from the schema; the result sets
  are compared.
* **valid_sql_rate** -- whether the predicted SQL parses at all.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import textwrap
from pathlib import Path
from typing import Any

import sqlparse
import typer
import yaml

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def execute_sql(sql: str, schema: str, timeout: int = 5) -> tuple[bool, list | str]:
    """Execute *sql* against an in-memory SQLite DB initialised with *schema*.

    Parameters
    ----------
    sql:
        The SQL query to run (SELECT, etc.).
    schema:
        One or more ``CREATE TABLE`` statements used to set up the database.
    timeout:
        Maximum seconds for the query.

    Returns
    -------
    (success, results_or_error)
        On success: ``(True, [rows...])``.
        On failure: ``(False, error_message)``.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA busy_timeout = %d" % (timeout * 1000))
        # Execute each statement in the schema separately
        for statement in sqlparse.split(schema):
            statement = statement.strip()
            if statement:
                conn.execute(statement)
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as exc:
        return False, str(exc)


def _normalise_sql(sql: str) -> str:
    """Lower-case, collapse whitespace, strip trailing semicolons."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def exact_match(predicted: str, reference: str) -> bool:
    """Normalised string comparison of two SQL queries."""
    return _normalise_sql(predicted) == _normalise_sql(reference)


def valid_sql(sql: str) -> bool:
    """Return ``True`` if *sql* parses as valid SQL via sqlparse."""
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and parsed[0].get_type() is not None
    except Exception:
        return False


def execution_match(predicted: str, reference: str, schema: str, timeout: int = 5) -> bool:
    """Return ``True`` if both queries execute and produce the same result set.

    Results are compared as sorted lists of tuples so row ordering does not
    affect the outcome.
    """
    ok_pred, res_pred = execute_sql(predicted, schema, timeout=timeout)
    ok_ref, res_ref = execute_sql(reference, schema, timeout=timeout)

    if not ok_pred or not ok_ref:
        return False

    return sorted(res_pred) == sorted(res_ref)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    test_data: list[dict[str, Any]],
    config: dict,
) -> dict[str, float]:
    """Run all metrics on a test set and return aggregate scores.

    Parameters
    ----------
    model_path:
        Path to the merged model (or adapter directory).
    test_data:
        List of dicts, each with ``"messages"`` (system/user/assistant list).
    config:
        Parsed eval YAML config.

    Returns
    -------
    Dict with keys ``exact_match``, ``execution_accuracy``, ``valid_sql_rate``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    import torch

    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    max_new_tokens = config.get("max_new_tokens", 256)
    timeout = config.get("execution_eval", {}).get("timeout_seconds", 5)
    n_samples = min(config.get("eval_samples", 500), len(test_data))
    exec_enabled = config.get("execution_eval", {}).get("enabled", True)

    exact_scores: list[int] = []
    exec_scores: list[int] = []
    valid_scores: list[int] = []

    from rich.progress import track

    for sample in track(test_data[:n_samples], description="Evaluating"):
        messages = sample["messages"]
        # messages[0]=system, [1]=user, [2]=assistant (gold)
        gold_sql = messages[2]["content"]
        schema = messages[0]["content"].split("SCHEMA:\n", 1)[-1]

        prompt = pipe.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        predicted_sql = outputs[0]["generated_text"][len(prompt) :].strip()

        exact_scores.append(int(exact_match(predicted_sql, gold_sql)))
        valid_scores.append(int(valid_sql(predicted_sql)))

        if exec_enabled:
            exec_scores.append(
                int(execution_match(predicted_sql, gold_sql, schema, timeout=timeout))
            )

    results = {
        "exact_match": sum(exact_scores) / max(len(exact_scores), 1),
        "valid_sql_rate": sum(valid_scores) / max(len(valid_scores), 1),
    }
    if exec_enabled:
        results["execution_accuracy"] = sum(exec_scores) / max(len(exec_scores), 1)

    logger.info("Evaluation results: %s", results)

    # Log to W&B if available
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(results)
    except ImportError:
        pass

    return results


@app.command()
def main(
    model_path: str = typer.Option("./checkpoints/text2sql", "--model-path", "-m"),
    config: str = typer.Option("configs/eval.yaml", "--config", "-c"),
    test_file: str = typer.Option("test_dataset.json", "--test-file", "-t"),
) -> None:
    """CLI entry-point for evaluation."""
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    with open(config) as f:
        eval_config = yaml.safe_load(f)

    with open(test_file) as f:
        test_data = json.load(f)

    results = evaluate_model(model_path, test_data, eval_config)
    from rich import print as rprint

    rprint(results)


if __name__ == "__main__":
    app()

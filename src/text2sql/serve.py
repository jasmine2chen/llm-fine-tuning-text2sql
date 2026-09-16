"""Start a vLLM inference server for the fine-tuned model."""

from __future__ import annotations

import logging
import subprocess
import sys

import typer

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    model_path: str = typer.Argument(..., help="Path to the merged model directory"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    tensor_parallel_size: int = typer.Option(1, "--tp", help="Tensor-parallel GPUs"),
    max_model_len: int = typer.Option(2048, "--max-model-len"),
) -> None:
    """Launch a vLLM OpenAI-compatible server.

    Requires the ``[serve]`` extra: ``pip install '.[serve]'``.

    The server exposes ``/v1/chat/completions`` and ``/v1/completions`` so any
    OpenAI-compatible client can query it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
    ]

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()

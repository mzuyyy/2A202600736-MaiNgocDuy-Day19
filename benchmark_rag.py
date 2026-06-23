#!/usr/bin/env python3
from pathlib import Path

from graphrag.benchmark import run_benchmark
from graphrag.config import Settings


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    settings = Settings.from_env(project_dir / ".env", project_dir)
    run_benchmark(
        settings,
        project_dir / "dataset/triples/graphrag_edges_langextract.csv",
        project_dir / "benchmark_output",
    )


if __name__ == "__main__":
    main()

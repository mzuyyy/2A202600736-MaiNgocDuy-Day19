#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from graphrag.config import Settings
from graphrag.visualize import draw_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the local GraphRAG graph")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_output/graph.png"))
    parser.add_argument("--max-nodes", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    settings = Settings.from_env(args.env_file.resolve(), project_dir)
    output = args.output
    if not output.is_absolute():
        output = project_dir / output
    written = draw_graph(settings.graph_path, output, max_nodes=args.max_nodes)
    print(f"wrote={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

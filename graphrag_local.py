#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAIError
from pydantic import ValidationError

from graphrag.app import answer, check_runtime, ingest
from graphrag.config import MissingSettingError, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local GraphRAG with NetworkX")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check", help="Check dataset, graph path, and LLM")

    ingest_parser = subcommands.add_parser("ingest", help="Build the local graph")
    ingest_parser.add_argument("--max-docs", type=int)

    ask_parser = subcommands.add_parser("ask", help="Ask from stored graph facts")
    ask_parser.add_argument("question")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    settings = Settings.from_env(
        env_file=args.env_file.resolve(),
        project_dir=project_dir,
    )

    match args.command:
        case "check":
            exit_code, message = check_runtime(settings)
            print(message)
            return exit_code
        case "ingest":
            stored, failed, triples_path = ingest(settings, max_docs=args.max_docs)
            print(
                f"stored={stored} triples | failed_chunks={failed} | "
                f"triples={triples_path}"
            )
            return 0 if failed == 0 else 2
        case "ask":
            print(answer(settings, args.question))
            return 0
        case unreachable:
            raise AssertionError(f"Unsupported command: {unreachable}")


def run(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (
        ConnectionError,
        MissingSettingError,
        OpenAIError,
        OSError,
        ValidationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class LlmSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    base_url: str = Field(min_length=1)
    model: str = "btl-2"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm: LlmSettings
    dataset_dir: Path
    graph_path: Path
    triples_path: Path
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    @classmethod
    def from_env(cls, env_file: Path, project_dir: Path) -> Settings:
        file_values = {
            key: value
            for key, value in dotenv_values(env_file).items()
            if value is not None
        }

        def value(name: str, default: str | None = None) -> str:
            resolved = os.environ.get(name, file_values.get(name, default))
            if resolved is None or not resolved.strip():
                raise MissingSettingError(name=name, env_file=env_file)
            return resolved.strip()

        dataset_dir = Path(value("DATASET_DIR", "dataset"))
        if not dataset_dir.is_absolute():
            dataset_dir = project_dir / dataset_dir
        graph_path = Path(value("GRAPH_PATH", "graph.graphml"))
        if not graph_path.is_absolute():
            graph_path = project_dir / graph_path
        triples_path = Path(
            value("TRIPLES_PATH", "dataset/triples/graphrag_edges_langextract.csv")
        )
        if not triples_path.is_absolute():
            triples_path = project_dir / triples_path

        return cls(
            llm=LlmSettings(
                api_key=SecretStr(value("BTL_API_KEY")),
                base_url=value("BTL_API_BASE_URL"),
                model=value("BTL_MODEL", "btl-2"),
            ),
            dataset_dir=dataset_dir.resolve(),
            graph_path=graph_path.resolve(),
            triples_path=triples_path.resolve(),
            chunk_size=int(value("CHUNK_SIZE", "1200")),
            chunk_overlap=int(value("CHUNK_OVERLAP", "150")),
        )


class MissingSettingError(RuntimeError):
    def __init__(self, name: str, env_file: Path) -> None:
        self.name = name
        self.env_file = env_file
        super().__init__(f"Missing {name} in environment or {env_file}")

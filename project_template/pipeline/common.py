from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    features_dir: Path
    artifacts_dir: Path

    @staticmethod
    def from_repo(repo_root: str | Path) -> "Paths":
        root = Path(repo_root).resolve()
        pt = root / "project_template"
        return Paths(
            root=root,
            data_dir=pt / "data",
            features_dir=pt / "features",
            artifacts_dir=pt / "artifacts",
        )


def repo_root_from_this_file() -> Path:
    # project_template/pipeline/common.py -> repo root
    return Path(__file__).resolve().parents[2]


def require_columns(df, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}；当前列: {list(df.columns)}")


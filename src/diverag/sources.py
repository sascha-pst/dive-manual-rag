"""Source manifest and license gate. Nothing is parsed unless its license is allowed."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ALLOWED_LICENSES = {"public-release", "us-gov-work", "licensed-written"}


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    publisher: str
    license: str
    revision: str
    effective: str
    url: str
    path: str


class LicenseError(RuntimeError):
    pass


def load_manifest(path: str | Path = "sources.toml") -> list[Source]:
    data = tomllib.loads(Path(path).read_text())
    sources = [Source(**s) for s in data["source"]]
    blocked = [s.id for s in sources if s.license not in ALLOWED_LICENSES]
    if blocked:
        raise LicenseError(f"Sources without an allowed license: {blocked}")
    return sources

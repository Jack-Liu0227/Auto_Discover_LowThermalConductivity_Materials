from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatabaseRecord:
    material_id: str
    formula: str
    source_url: str | None = None
    symmetry: str | None = None
    stability: float | None = None
    band_gap: float | None = None
    energy_above_hull: float | None = None
    cif_content: str | None = None
    cif_path: str | None = None
    structure_available: bool | None = None
    structure_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatabaseQueryResult:
    success: bool
    database: str
    query: dict[str, Any]
    records: list[DatabaseRecord] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "database": self.database,
            "query": self.query,
            "records": [r.to_dict() for r in self.records],
            "error": self.error,
        }

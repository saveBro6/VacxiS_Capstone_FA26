"""Discover markdown files and resolve vaccine names from folder structure."""
import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import slugify

# Scanned base directories
SOURCE_DIRS = [
    "CLEANED/cleaned_nghia",
    "CLEANED/HIEU/hieu",
    "CLEANED/HUNG",
    "CLEANED/MANH/MANH(HIEU+HUNG_LAM)",
]


@dataclass
class VaccineDocument:
    """Represents a discovered vaccine document."""
    folder_id: str
    vaccine_name: str
    md_files: list[Path]
    source_dir: str

    @property
    def doc_id(self) -> str:
        """Generate doc_id from slugified vaccine name and folder ID."""
        return slugify(f"{self.vaccine_name}_{self.folder_id}")

    @property
    def doc_title(self) -> str:
        """Generate human-readable document title."""
        return f"Hướng dẫn sử dụng Vaccine {self.vaccine_name}"


def discover_documents(
    project_root: Path,
    mapping_path: Path | None = None,
) -> list[VaccineDocument]:
    """Scan CLEANED directories and return list of discovered documents."""
    if mapping_path is None:
        mapping_path = project_root / "75_vaccine_names_mapping.json"
        if not mapping_path.exists():
            mapping_path = project_root / "vaccine_name_mapping.json"

    with open(mapping_path, "r", encoding="utf-8") as f:
        name_mapping: dict[str, str] = json.load(f)

    documents: list[VaccineDocument] = []

    for src_dir_rel in SOURCE_DIRS:
        src_dir = project_root / src_dir_rel
        if not src_dir.exists():
            continue

        for folder in sorted(src_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_id = folder.name
            md_files = sorted(folder.glob("*.md"))

            if not md_files:
                continue

            vaccine_name = name_mapping.get(folder_id, folder_id)

            documents.append(
                VaccineDocument(
                    folder_id=folder_id,
                    vaccine_name=vaccine_name,
                    md_files=md_files,
                    source_dir=src_dir_rel,
                )
            )

    return documents

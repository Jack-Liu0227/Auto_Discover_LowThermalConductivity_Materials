"""Small Markdown document reader used by the evaluation agents."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


class DocumentReader:
    """Load a Markdown document and expose sections, metadata, and search."""

    def __init__(self, doc_path: str):
        self.doc_path = doc_path
        self.content = ""
        self.sections: Dict[str, str] = {}
        self.metadata: Dict[str, str] = {}

        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")
        self._load_document()
        self._parse_sections()

    def _load_document(self) -> None:
        with open(self.doc_path, "r", encoding="utf-8") as handle:
            self.content = handle.read()

    def _parse_sections(self) -> None:
        metadata_patterns = {
            "version": r"\*\*(?:Version|版本)\*\*\s*:\s*(.+)",
            "date": r"\*\*(?:Date|日期)\*\*\s*:\s*(.+)",
        }
        for name, pattern in metadata_patterns.items():
            match = re.search(pattern, self.content, flags=re.IGNORECASE)
            if match:
                self.metadata[name] = match.group(1).strip()

        sections = re.split(r"^##\s+", self.content, flags=re.MULTILINE)
        for section in sections[1:]:
            title, separator, body = section.partition("\n")
            if separator:
                self.sections[title.strip()] = body.strip()

    def get_section(self, section_title: str) -> Optional[str]:
        if section_title in self.sections:
            return self.sections[section_title]
        query = section_title.casefold()
        for title, content in self.sections.items():
            if query in title.casefold():
                return content
        return None

    def list_sections(self) -> List[str]:
        return list(self.sections)

    def get_full_content(self) -> str:
        return self.content

    def get_metadata(self) -> Dict[str, str]:
        return dict(self.metadata)

    def search(self, keyword: str, context_lines: int = 3) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        query = keyword.casefold()
        for section_title, section_content in self.sections.items():
            lines = section_content.splitlines()
            for index, line in enumerate(lines):
                if query not in line.casefold():
                    continue
                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                results.append(
                    {
                        "section": section_title,
                        "match": line.strip(),
                        "context": "\n".join(lines[start:end]),
                    }
                )
        return results

    def get_summary(self, max_length: int = 500) -> str:
        if max_length <= 0:
            return ""
        summary: List[str] = []
        length = 0
        for line in self.content.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            summary.append(line)
            length += len(line)
            if length >= max_length or len(summary) >= 5:
                break
        return "\n".join(summary)

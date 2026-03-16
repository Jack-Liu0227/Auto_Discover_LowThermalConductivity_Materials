from __future__ import annotations

import re


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def extract_markdown_section(doc_text: str, heading_prefix: str) -> str:
    lines = (doc_text or "").splitlines()
    start_idx: int | None = None
    start_level: int | None = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        match = HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if title.startswith(heading_prefix):
            start_idx = idx
            start_level = level
            break

    if start_idx is None or start_level is None:
        return ""

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx].strip()
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) <= start_level:
            end_idx = idx
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def compact_markdown_for_query(text: str, max_chars: int = 600) -> str:
    snippets: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        line = line.lstrip("#").strip()
        line = line.lstrip(">").strip()
        line = line.replace("**", "").replace("`", "")
        line = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", line)
        line = re.sub(r"\s+", " ", line)
        if line:
            snippets.append(line)

    compact = " | ".join(snippets)
    compact = re.sub(r"\s+", " ", compact).strip(" |")
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def build_websearch_theory_context(doc_text: str, max_chars: int = 600) -> str:
    dynamic_section = extract_markdown_section(doc_text, "4.")
    workflow_prior = extract_markdown_section(doc_text, "3.4")
    design_theory = extract_markdown_section(doc_text, "2.")

    parts = [part for part in (dynamic_section, workflow_prior, design_theory) if part]
    if not parts:
        return ""

    return compact_markdown_for_query("\n\n".join(parts), max_chars=max_chars)

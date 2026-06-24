from __future__ import annotations

AGNO_STATE_KEYS = {
    "all_results": "all_results",
    "last_result": "last_result",
    "last_iteration": "last_iteration",
    "candidate_top20": "candidate_top20",
    "screened_top10": "screened_top10",
    "calculation_results": "calculation_results",
    "extraction_result": "extraction_result",
    "updated_data_path": "updated_data_path",
    "updated_doc_path": "updated_doc_path",
    "errors": "errors",
}

AGNO_SESSION_STATE_DEFAULT = {
    AGNO_STATE_KEYS["all_results"]: [],
    AGNO_STATE_KEYS["last_result"]: {},
    AGNO_STATE_KEYS["last_iteration"]: 0,
    AGNO_STATE_KEYS["candidate_top20"]: [],
    AGNO_STATE_KEYS["screened_top10"]: [],
    AGNO_STATE_KEYS["calculation_results"]: {},
    AGNO_STATE_KEYS["extraction_result"]: {},
    AGNO_STATE_KEYS["updated_data_path"]: None,
    AGNO_STATE_KEYS["updated_doc_path"]: None,
    AGNO_STATE_KEYS["errors"]: [],
}

__all__ = ["AGNO_STATE_KEYS", "AGNO_SESSION_STATE_DEFAULT"]

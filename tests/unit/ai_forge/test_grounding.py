"""Tests for AI Forge RAG grounding."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forgequant.ai_forge.exceptions import GroundingError
from forgequant.ai_forge.grounding import (
    KnowledgeDocument,
    load_all_documents,
    load_documents_from_json,
)


def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestLoadDocumentsFromJson:
    def test_valid_file(self, tmp_path: Path) -> None:
        docs = [
            {"id": "1", "title": "Test", "content": "Test content"},
            {"id": "2", "title": "Test 2", "content": "More content"},
        ]
        path = tmp_path / "test.json"
        _write_json(path, docs)

        result = load_documents_from_json(path)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["content"] == "More content"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(GroundingError, match="not found"):
            load_documents_from_json(Path("/nonexistent/file.json"))

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(GroundingError, match="Invalid JSON"):
            load_documents_from_json(path)

    def test_not_array_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        _write_json(path, {"key": "value"})
        with pytest.raises(GroundingError, match="JSON array"):
            load_documents_from_json(path)

    def test_missing_required_fields_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"
        _write_json(path, [{"id": "1", "title": "No content"}])
        with pytest.raises(GroundingError, match="missing required"):
            load_documents_from_json(path)

    def test_non_dict_element_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_elem.json"
        _write_json(path, ["not a dict"])
        with pytest.raises(GroundingError, match="not an object"):
            load_documents_from_json(path)

    def test_with_optional_fields(self, tmp_path: Path) -> None:
        docs = [
            {
                "id": "1",
                "title": "Full doc",
                "content": "Content here",
                "category": "strategy",
                "tags": ["a", "b"],
            }
        ]
        path = tmp_path / "full.json"
        _write_json(path, docs)
        result = load_documents_from_json(path)
        assert result[0]["category"] == "strategy"


class TestLoadAllDocuments:
    def test_loads_multiple_files(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_json(
                tmp_path / f"doc{i}.json",
                [{"id": f"doc{i}", "title": f"Doc {i}", "content": f"Content {i}"}],
            )
        result = load_all_documents(tmp_path)
        assert len(result) == 3

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = load_all_documents(tmp_path)
        assert result == []

    def test_nonexistent_directory_raises(self) -> None:
        with pytest.raises(GroundingError, match="not found"):
            load_all_documents(Path("/nonexistent/dir"))

    def test_not_directory_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.touch()
        with pytest.raises(GroundingError, match="Not a directory"):
            load_all_documents(file_path)

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / "valid.json",
            [{"id": "1", "title": "T", "content": "C"}],
        )
        (tmp_path / "readme.txt").write_text("not json", encoding="utf-8")
        result = load_all_documents(tmp_path)
        assert len(result) == 1


class TestKnowledgeBaseDocuments:
    def test_trading_concepts_valid(self) -> None:
        kb_dir = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "forgequant"
            / "knowledge_base"
            / "documents"
        )
        if not kb_dir.exists():
            pytest.skip("Knowledge base documents directory not found")

        concepts_path = kb_dir / "trading_concepts.json"
        if concepts_path.exists():
            docs = load_documents_from_json(concepts_path)
            assert len(docs) >= 1
            for doc in docs:
                assert len(doc["content"]) > 50

    def test_block_catalog_valid(self) -> None:
        kb_dir = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "forgequant"
            / "knowledge_base"
            / "documents"
        )
        if not kb_dir.exists():
            pytest.skip("Knowledge base documents directory not found")

        catalog_path = kb_dir / "block_catalog.json"
        if catalog_path.exists():
            docs = load_documents_from_json(catalog_path)
            assert len(docs) >= 1

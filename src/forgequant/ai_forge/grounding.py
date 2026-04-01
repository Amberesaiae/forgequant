"""
RAG grounding using ChromaDB.

Provides knowledge base ingestion and retrieval for enriching
the LLM system prompt with relevant trading domain knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forgequant.ai_forge.exceptions import GroundingError
from forgequant.core.logging import get_logger

logger = get_logger(__name__)

KnowledgeDocument = dict[str, Any]


def load_documents_from_json(file_path: Path) -> list[KnowledgeDocument]:
    if not file_path.exists():
        raise GroundingError("load", f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise GroundingError("load", f"Invalid JSON in {file_path}: {e}") from e
    except OSError as e:
        raise GroundingError("load", f"Cannot read {file_path}: {e}") from e

    if not isinstance(data, list):
        raise GroundingError(
            "load",
            f"Expected a JSON array in {file_path}, got {type(data).__name__}",
        )

    validated: list[KnowledgeDocument] = []
    for i, doc in enumerate(data):
        if not isinstance(doc, dict):
            raise GroundingError(
                "load",
                f"Document at index {i} is not an object: {type(doc).__name__}",
            )

        required_fields = {"id", "title", "content"}
        missing = required_fields - set(doc.keys())
        if missing:
            raise GroundingError(
                "load",
                f"Document at index {i} missing required fields: {missing}",
            )

        validated.append(doc)

    logger.info("documents_loaded", file=str(file_path), count=len(validated))
    return validated


def load_all_documents(directory: Path) -> list[KnowledgeDocument]:
    if not directory.exists():
        raise GroundingError("load_all", f"Directory not found: {directory}")

    if not directory.is_dir():
        raise GroundingError("load_all", f"Not a directory: {directory}")

    all_docs: list[KnowledgeDocument] = []
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        logger.warning("no_json_files_found", directory=str(directory))
        return all_docs

    for file_path in json_files:
        docs = load_documents_from_json(file_path)
        all_docs.extend(docs)

    logger.info(
        "all_documents_loaded",
        directory=str(directory),
        file_count=len(json_files),
        total_docs=len(all_docs),
    )

    return all_docs


class KnowledgeBase:
    """ChromaDB-backed knowledge base for RAG grounding."""

    def __init__(
        self,
        persist_directory: str | Path = "./data/chromadb",
        collection_name: str = "trading_knowledge",
    ) -> None:
        self._persist_dir = str(persist_directory)
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_initialized(self) -> None:
        if self._client is not None:
            return

        try:
            import chromadb
        except ImportError as e:
            raise GroundingError(
                "init",
                "chromadb is not installed. Install with: pip install forgequant[ai]",
            ) from e

        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "ForgeQuant trading knowledge base"},
            )
            logger.info(
                "chromadb_initialized",
                persist_dir=self._persist_dir,
                collection=self._collection_name,
                existing_count=self._collection.count(),
            )
        except Exception as e:
            raise GroundingError("init", f"ChromaDB initialization failed: {e}") from e

    def ingest_documents(
        self,
        documents: list[KnowledgeDocument],
        batch_size: int = 100,
    ) -> int:
        self._ensure_initialized()

        if not documents:
            return 0

        total = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            ids: list[str] = []
            contents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for doc in batch:
                doc_id = str(doc["id"])
                content = doc["content"]
                metadata: dict[str, Any] = {
                    "title": doc.get("title", ""),
                    "category": doc.get("category", "general"),
                }

                tags = doc.get("tags", [])
                if isinstance(tags, list):
                    metadata["tags"] = ",".join(str(t) for t in tags)

                ids.append(doc_id)
                contents.append(content)
                metadatas.append(metadata)

            try:
                self._collection.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                )
                total += len(batch)
            except Exception as e:
                raise GroundingError(
                    "ingest",
                    f"Failed to upsert batch starting at index {i}: {e}",
                ) from e

        logger.info("documents_ingested", count=total)
        return total

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
    ) -> str:
        self._ensure_initialized()

        if not query.strip():
            return ""

        try:
            where_filter = None
            if category_filter:
                where_filter = {"category": category_filter}

            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count() or 1),
                where=where_filter,
            )
        except Exception as e:
            raise GroundingError("retrieve", f"Query failed: {e}") from e

        if not results or not results.get("documents") or not results["documents"][0]:
            return ""

        sections: list[str] = []
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            title = metadata.get("title", f"Document {i + 1}")
            relevance = 1.0 - (distances[i] if i < len(distances) else 0.0)

            sections.append(
                f"--- {title} (relevance: {relevance:.2f}) ---\n{doc_text}"
            )

        context = "\n\n".join(sections)

        logger.debug(
            "rag_retrieval_complete",
            query_length=len(query),
            results_count=len(documents),
        )

        return context

    def count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()

    def clear(self) -> None:
        self._ensure_initialized()

        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
            )
            logger.warning("knowledge_base_cleared", collection=self._collection_name)
        except Exception as e:
            raise GroundingError("clear", f"Failed to clear collection: {e}") from e

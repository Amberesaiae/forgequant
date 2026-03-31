"""
RAG Knowledge Base for AI Forge.

Loads trading knowledge into ChromaDB for grounding AI decisions.
"""

import json
from pathlib import Path
from typing import List, Optional

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


def load_knowledge_base(
    knowledge_path: Optional[Path] = None,
    persist_path: Optional[str] = None,
) -> Optional["chromadb.Collection"]:
    """Load trading knowledge into ChromaDB for RAG.

    Args:
        knowledge_path: Path to knowledge base JSON files.
        persist_path: Path to persist ChromaDB.

    Returns:
        ChromaDB collection or None if chromadb is not installed.
    """
    if not HAS_CHROMA:
        return None

    if persist_path is None:
        persist_path = str(Path(__file__).parent.parent.parent / "knowledge_base" / "chroma")

    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name="trading_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    if knowledge_path and knowledge_path.exists():
        _ingest_documents(collection, knowledge_path)

    return collection


def _ingest_documents(
    collection: "chromadb.Collection",
    knowledge_path: Path,
) -> None:
    """Ingest knowledge documents into ChromaDB.

    Args:
        collection: ChromaDB collection.
        knowledge_path: Path to knowledge base directory.
    """
    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[dict] = []

    for json_file in knowledge_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        doc_id = json_file.stem
        text = data.get("text", "")
        metadata = data.get("metadata", {"source": json_file.name})

        if text:
            documents.append(text)
            ids.append(doc_id)
            metadatas.append(metadata)

    if documents:
        # Check what's already in the collection
        existing = collection.get(ids=ids)
        if existing and existing["ids"]:
            collection.update(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
        else:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

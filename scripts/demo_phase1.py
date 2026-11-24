"""Phase 1 demo: ingest, dedupe, list, delete using the real embedding model."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.store import DocumentStore, make_embeddings  # noqa: E402

PDF = ROOT / "data" / "sample.pdf"


def main() -> None:
    """Run the demo against a temporary index directory."""
    store = DocumentStore(make_embeddings(), tempfile.mkdtemp(prefix="citerag_demo_"))
    r = store.add_document(PDF)
    print(f"ingest: doc_id={r.doc_id} chunks={r.chunks} skipped={r.skipped}")
    r2 = store.add_document(PDF)
    print(f"re-ingest: doc_id={r2.doc_id} chunks={r2.chunks} skipped={r2.skipped}")
    print("list:", store.list_documents())
    for doc, score in store.similarity_search("What does error E-4012 mean?", 2):
        print(f"search hit: {doc.metadata['chunk_id']} score={score:.3f}")
    print("delete:", store.delete_document(r.doc_id))
    print("list:", store.list_documents(), "chunks left:", len(store.all_chunks()))


if __name__ == "__main__":
    main()

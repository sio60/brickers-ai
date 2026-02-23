# vectordb
from vectordb.loader import ingest_parts, ingest_models, ensure_indexes
from vectordb.processor import update_all_bboxes, update_all_embeddings
from vectordb.resolver import resolve_part, parts_vector_search, get_bbox_doc
from vectordb.maintenance import run_full_sync, start_scheduler
from vectordb.utils import seed_dummy_parts

__all__ = [
    "ingest_parts", "ingest_models", "ensure_indexes",
    "update_all_bboxes", "update_all_embeddings",
    "resolve_part", "parts_vector_search", "get_bbox_doc",
    "run_full_sync", "start_scheduler", "seed_dummy_parts"
]

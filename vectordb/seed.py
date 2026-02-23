# vectordb/seed.py
import config
from db import get_parts_collection
from vectordb.utils import seed_dummy_parts

if __name__ == "__main__":
    col = get_parts_collection()
    n = seed_dummy_parts(col, config.EMBEDDING_DIMS, config.VECTOR_FIELD, overwrite=True)
    print(f"Inserted {n} dummy docs into {config.PARTS_COLLECTION}")

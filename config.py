# config.py
# アプリ全体で共通の定数をここで一元管理します。
# app.py / ingest.py / query.py はすべてここから読み込みます。

DOCS_DIR: str = "docs"
INDEX_DIR: str = "faiss_index"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ingest パラメータ
CHUNK_SIZE: int = 800       # 日本語文書向けに調整（500→800）
CHUNK_OVERLAP: int = 100

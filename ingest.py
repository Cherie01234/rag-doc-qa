# ingest.py
import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # fix: langchain_community は非推奨

from config import DOCS_DIR, INDEX_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()


def load_documents():
    documents = []
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    for filename in os.listdir(DOCS_DIR):
        full_path = os.path.join(DOCS_DIR, filename)
        try:
            if filename.lower().endswith(".pdf"):
                docs = PyPDFLoader(full_path).load()
            elif filename.lower().endswith(".txt"):
                docs = TextLoader(full_path, encoding="utf-8").load()
            else:
                continue

            # source をファイル名で正規化
            for d in docs:
                d.metadata["source"] = filename
            documents.extend(docs)

        except (OSError, FileNotFoundError) as e:
            # ファイルアクセス系エラー（パーミッション・破損など）
            print(f"[WARN] ファイルアクセスエラーのためスキップ: {filename} - {e}")
        except Exception as e:
            # PDF パース失敗など予期しないエラー
            print(f"[WARN] 読み込みエラーのためスキップ: {filename} - {e}")

    return documents


def run_ingest():
    """
    UI やコマンドから呼べるインデックス再作成関数。
    戻り値: (num_chunks, num_documents)
    """
    documents = load_documents()
    if not documents:
        print("インデックス対象のドキュメントがありません")
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    num_docs = len(set(d.metadata["source"] for d in chunks))
    print(f"Indexed {len(chunks)} chunks from {num_docs} documents.")
    return len(chunks), num_docs


if __name__ == "__main__":
    run_ingest()

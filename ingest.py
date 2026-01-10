# ingest.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

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

            # sourceをファイル名で正規化
            for d in docs:
                d.metadata["source"] = filename

            documents.extend(docs)

        except Exception as e:
            print(f"[WARN] {filename} をスキップしました: {e}")

    return documents

def run_ingest():
    """
    UIやコマンドから呼べるインデックス再作成関数。
    戻り値: (num_chunks, num_documents)
    """
    documents = load_documents()

    if not documents:
        print("インデックス対象のドキュメントがありません")
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    num_docs = len(set(d.metadata["source"] for d in chunks))
    print(f"Indexed {len(chunks)} chunks from {num_docs} documents.")
    return len(chunks), num_docs

if __name__ == "__main__":
    run_ingest()

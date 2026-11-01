# eval/run_eval.py  ― 検索（retrieval）精度の評価（有料APIは使わない）
#
# eval/corpus の文書を ingest.py と同じ方法で分割・埋め込みして FAISS に登録し、
# eval/questions.jsonl の各質問で similarity_search した結果から Hit@k と MRR を計算する。
# 結果は eval/results.json に保存する。
#
#   python eval/run_eval.py                 # 全条件（埋め込みモデル × チャンクサイズ）
#   python eval/run_eval.py --chunk-sizes 800 --models all-MiniLM-L6-v2
import argparse
import json
import sys
from collections import Counter
from datetime import date
from math import comb
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))
import config  # noqa: E402

CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_PATH = EVAL_DIR / "results.json"

# 比較する埋め込みモデル。先頭はアプリと同じ設定（config.EMBEDDING_MODEL・既定の引数）。
# multilingual-e5 は学習時と同じく、質問に "query: "、文書に "passage: " を付けて埋め込む。
MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": config.EMBEDDING_MODEL,
        "encode_kwargs": {},
        "query_encode_kwargs": {},
    },
    "multilingual-e5-small": {
        "model_name": "intfloat/multilingual-e5-small",
        "encode_kwargs": {"prompt": "passage: ", "normalize_embeddings": True},
        "query_encode_kwargs": {"prompt": "query: ", "normalize_embeddings": True},
    },
}
CHUNK_SIZES = [300, 500, config.CHUNK_SIZE]
K_VALUES = [1, 3, 5, 10]


# =========================
# データ
# =========================

def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    # ingest.py の TextLoader と同じく、ファイル全体を1文書として読み、source はファイル名にする
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(corpus_dir.glob("*.txt"))
    ]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    # ingest.py と同じ分割器。結果を確かめやすいよう「ファイル名#通し番号」の chunk_id を付ける
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    counts = Counter()
    for c in chunks:
        source = c.metadata["source"]
        c.metadata["chunk_id"] = f"{source}#{counts[source]}"
        counts[source] += 1
    return chunks


# =========================
# 指標
# =========================

def is_relevant(chunk: Document, question: dict) -> bool:
    """正解の文書から取られ、正解の根拠文（evidence）を丸ごと含むチャンクを正解とみなす。"""
    return chunk.metadata.get("source") == question["doc"] and question["evidence"] in chunk.page_content


def first_relevant_rank(retrieved: list[Document], question: dict) -> int | None:
    """検索結果のうち最初の正解チャンクの順位（1始まり）。上位に無ければ None。"""
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, question):
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def random_hit_at_k(n_chunks: int, n_relevant: int, k: int) -> float:
    """チャンクを無作為に並べたときの Hit@k の期待値（比較の目安）。"""
    k = min(k, n_chunks)
    return 1 - comb(n_chunks - n_relevant, k) / comb(n_chunks, k)


# =========================
# 評価
# =========================

def make_embeddings(model_key: str, device: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    spec = MODELS[model_key]
    return HuggingFaceEmbeddings(
        model_name=spec["model_name"],
        model_kwargs={"device": device},
        encode_kwargs=spec["encode_kwargs"],
        query_encode_kwargs=spec["query_encode_kwargs"],
    )


def evaluate(embeddings, chunks: list[Document], questions: list[dict]) -> dict:
    from langchain_community.vectorstores import FAISS

    # アプリの「全PDF横断」と同じく、source で絞り込まずに全チャンクから検索する
    vectorstore = FAISS.from_documents(chunks, embeddings)
    max_k = max(K_VALUES)

    ranks = []
    n_relevant = []
    top1 = Counter()
    for q in questions:
        retrieved = vectorstore.similarity_search(q["question"], k=max_k)
        ranks.append(first_relevant_rank(retrieved, q))
        n_relevant.append(sum(1 for c in chunks if is_relevant(c, q)))
        top1[retrieved[0].metadata["chunk_id"]] += 1
    # 質問が違っても同じチャンクばかり1位になっていないか（埋め込みが質問を区別できているか）の目安
    top1_chunk, top1_count = top1.most_common(1)[0]

    metrics = {f"hit@{k}": round(hit_at_k(ranks, k), 4) for k in K_VALUES}
    metrics[f"mrr@{max_k}"] = round(mrr(ranks), 4)
    random_baseline = {
        f"hit@{k}": round(sum(random_hit_at_k(len(chunks), n, k) for n in n_relevant) / len(questions), 4)
        for k in K_VALUES
    }
    return {
        "metrics": metrics,
        "random_baseline": random_baseline,
        "most_common_top1": {"chunk_id": top1_chunk, "n_questions": top1_count},
        "ranks": {q["id"]: r for q, r in zip(questions, ranks)},
    }


def main():
    parser = argparse.ArgumentParser(description="検索精度（Hit@k・MRR）の評価")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=CHUNK_SIZES)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--device", default="cpu", help="埋め込みの計算に使う device（再現性のため既定は cpu）")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    documents = load_corpus()
    questions = load_questions()

    runs = []
    for model_key in args.models:
        embeddings = make_embeddings(model_key, args.device)
        for chunk_size in args.chunk_sizes:
            chunks = split_documents(documents, chunk_size, args.chunk_overlap)
            result = evaluate(embeddings, chunks, questions)
            runs.append({
                "embedding_model": MODELS[model_key]["model_name"],
                "chunk_size": chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "n_chunks": len(chunks),
                "is_app_setting": (
                    MODELS[model_key]["model_name"] == config.EMBEDDING_MODEL
                    and chunk_size == config.CHUNK_SIZE
                    and args.chunk_overlap == config.CHUNK_OVERLAP
                ),
                **result,
            })
            m = result["metrics"]
            print(f"{model_key:24s} chunk={chunk_size:4d} n_chunks={len(chunks):3d}  "
                  + "  ".join(f"{name}={value:.3f}" for name, value in m.items()))

    from importlib.metadata import version

    results = {
        "generated_on": date.today().isoformat(),
        "dataset": {
            "n_documents": len(documents),
            "n_questions": len(questions),
            "corpus_chars": sum(len(d.page_content) for d in documents),
        },
        "settings": {
            "search": "FAISS similarity_search（sourceの絞り込みなし＝アプリの全PDF横断と同じ）",
            "relevance": "正解文書のチャンクのうち、根拠文（evidence）を丸ごと含むもの",
            "k_values": K_VALUES,
            "device": args.device,
        },
        "environment": {
            "python": sys.version.split()[0],
            **{pkg: version(pkg) for pkg in [
                "sentence-transformers", "faiss-cpu", "langchain-community",
                "langchain-huggingface", "langchain-text-splitters", "torch",
            ]},
        },
        "runs": runs,
    }
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

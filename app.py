# app.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import time

from ingest import run_ingest  # ingest.py に run_ingest() がある前提

load_dotenv()

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

st.set_page_config(page_title="PDF管理付き RAG UI", layout="wide")
st.title("📚 PDF管理付き RAG（OpenRouter対応）")

# =========================
# 再実行ヘルパー（互換対応）
# =========================
def force_rerun():
    """
    Streamlit の再実行を安全に行うヘルパー。
    - 可能なら st.experimental_rerun() を使う
    - なければ query params を更新して強制再実行を誘発する
    """
    try:
        # 互換性のある場合はこれで即再実行
        st.rerun()
    except Exception:
        # 一部の Streamlit では experimental_rerun が存在しないためフォールバック
        try:
            st.query_params(_rerun=int(time.time()))
        except Exception:
            # 最終手段：ユーザーにリロードを促すメッセージを出して処理停止
            st.info("ページを再読み込みしてください（自動リロードに失敗しました）。")
            st.stop()

# =========================
# FAISSロード（キャッシュ付き）
# =========================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    try:
        vs = FAISS.load_local(
            INDEX_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vs
    except Exception as e:
        print(f"[INFO] FAISS load failed: {e}")
        return None

# =========================
# LLM ローダ（簡易）
# =========================
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

# =========================
# プロンプト
# =========================
QA_PROMPT = ChatPromptTemplate.from_template(
    """以下は参考情報です。
---------
{context}
---------

質問:
{question}

日本語で簡潔かつ正確に答えてください。
"""
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """以下の文書を日本語で要約してください。
---------
{context}
---------
"""
)

# Ensure docs dir exists
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# =========================
# サイドバー：PDF管理（アップロードはボタン実行方式）
# =========================
st.sidebar.header("PDF 管理")

# アップロード（ファイル選択）
uploaded_files = st.sidebar.file_uploader(
    "PDFを選択（複数可）",
    type="pdf",
    accept_multiple_files=True,
    key="uploader"
)

# 選択されたファイル名のプレビュー
if uploaded_files:
    st.sidebar.markdown("**選択ファイル**")
    for f in uploaded_files:
        st.sidebar.write(f.name)

# アップロード保存の確定ボタン（重要：ここでのみ保存）
if st.sidebar.button("アップロードを実行") and uploaded_files:
    added = []
    for uploaded in uploaded_files:
        save_name = uploaded.name
        save_path = os.path.join(DOCS_DIR, save_name)
        # 衝突回避: 同名が既にある場合、タイムスタンプ付与（上書きしたいなら変える）
        if os.path.exists(save_path):
            base, ext = os.path.splitext(save_name)
            save_name = f"{base}_{int(time.time())}{ext}"
            save_path = os.path.join(DOCS_DIR, save_name)
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            added.append(save_name)
        except Exception as e:
            st.sidebar.error(f"{uploaded.name} の保存失敗: {e}")
    if added:
        st.sidebar.success(f"追加: {', '.join(added)}")
        # キャッシュクリアして一覧を最新化
        load_vectorstore.clear()
        force_rerun()

# =========================
# 現在の docs フォルダのファイル表示（ファイルシステムベース）
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("**現在の docs フォルダ（ファイルシステム）**")
pdf_files_fs = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])
if pdf_files_fs:
    for p in pdf_files_fs:
        st.sidebar.write(f"- {p}")
else:
    st.sidebar.write("_docs フォルダにPDFがありません_")

# 削除（ユーザーが選択して削除ボタンを押す方式）
delete_targets = st.sidebar.multiselect("削除するファイルを選択", pdf_files_fs, key="del")
if st.sidebar.button("選択ファイルを削除"):
    if delete_targets:
        deleted = []
        for tgt in delete_targets:
            try:
                os.remove(os.path.join(DOCS_DIR, tgt))
                deleted.append(tgt)
            except Exception as e:
                st.sidebar.error(f"{tgt} の削除に失敗: {e}")
        if deleted:
            st.sidebar.success(f"削除: {', '.join(deleted)}")
            load_vectorstore.clear()
            # 再起動してサイドバーを最新化
            force_rerun()
    else:
        st.sidebar.info("削除対象を選んでください")

st.sidebar.markdown("---")

# インデックス再構築（ingest.run_ingest）
if st.sidebar.button("インデックス再構築 (ingest.py 実行)"):
    # 注意: st.sidebar.spinner は無い -> st.spinner を使う
    with st.spinner("インデックス作成中...（時間がかかる場合があります）"):
        chunks, docs_count = run_ingest()
        st.sidebar.success(f"Indexed {chunks} chunks from {docs_count} documents.")
        load_vectorstore.clear()
        force_rerun()

st.sidebar.markdown("---")
st.sidebar.caption("※ PDF追加後はインデックス再構築を行ってください")

# =========================
# メインUI：検索 / 要約（インデックスベースの一覧も表示）
# =========================
mode = st.radio("検索モード", ["全PDF横断", "単一PDF"], horizontal=True)

# vectorstore 読み込み（キャッシュ）
vectorstore = load_vectorstore()

# 現在インデックスに含まれるPDF一覧（インデックスベース）
indexed_pdfs = []
if vectorstore is not None:
    try:
        indexed_pdfs = sorted({d.metadata.get("source") for d in vectorstore.docstore._dict.values()})
    except Exception:
        indexed_pdfs = []

st.markdown("**現在インデックスに含まれるPDF**")
if indexed_pdfs:
    for p in indexed_pdfs:
        st.write(f"- {p}")
else:
    st.write("_インデックスがありません_")

# 単一PDF選択（インデックスベースの選択肢）
selected_pdf = None
if mode == "単一PDF":
    if indexed_pdfs:
        selected_pdf = st.selectbox("対象PDFを選択", indexed_pdfs)
    else:
        st.warning("インデックスが存在しないため、単一PDFモードは選べません。まずインデックスを作成してください。")

question = st.text_area("質問（空欄の場合は要約）", value="", height=120)

if st.button("実行"):
    with st.spinner("検索中..."):
        vectorstore = load_vectorstore()
        if vectorstore is None:
            st.warning("インデックスが存在しません。まずサイドバーの「インデックス再構築」を実行してください。")
            st.stop()

        search_query = question.strip() if question.strip() else "この文書の全体概要 要点 まとめ"

        if mode == "単一PDF":
            docs = vectorstore.similarity_search(search_query, k=8, filter={"source": selected_pdf})
        else:
            docs = vectorstore.similarity_search(search_query, k=12)

        if not docs:
            st.warning("該当する情報が見つかりませんでした")
            st.stop()

        if mode == "全PDF横断":
            referenced_pdfs = sorted({d.metadata.get("source", "unknown") for d in docs})
            st.subheader("📂 参照されたPDF")
            for pdf in referenced_pdfs:
                st.write(f"- {pdf}")

        context = "\n\n".join(d.page_content for d in docs)
        if question.strip():
            prompt = QA_PROMPT.format(context=context, question=question)
        else:
            prompt = SUMMARY_PROMPT.format(context=context)

        llm = load_llm()
        response = llm.invoke([HumanMessage(content=prompt)])

        st.subheader("📝 回答")
        st.write(response.content)

        with st.expander("🔍 参照された文書チャンク"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**[{i}] {d.metadata.get('source')}**")
                st.write(d.page_content)

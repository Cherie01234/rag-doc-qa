import os
import re
import sys
import html
import subprocess
from collections import Counter
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # fix: langchain_community は非推奨
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import DOCS_DIR, INDEX_DIR, EMBEDDING_MODEL

# =========================
# 初期設定
# =========================
load_dotenv()

st.set_page_config(page_title="OpenRouter対応RAGドキュメントQA", layout="wide")
st.title("📄 OpenRouter対応RAG ドキュメントQA（Streamlit）")

# =========================
# Prompt
# =========================
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは与えられた文書だけを根拠に回答するアシスタントです。推測で断定しないでください。"),
        ("human", "【文書】\n{context}\n\n【質問】\n{question}\n\n日本語で簡潔に回答してください。"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは与えられた文書だけを根拠に要約するアシスタントです。推測で断定しないでください。"),
        ("human", "【文書】\n{context}\n\n日本語で要点を箇条書きで要約してください。"),
    ]
)

# =========================
# Utility
# =========================

def ensure_dirs():
    os.makedirs(DOCS_DIR, exist_ok=True)


def list_pdfs() -> List[str]:
    if not os.path.exists(DOCS_DIR):
        return []
    return sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")])


def run_ingest() -> bool:
    """
    可能なら ingest.py の run_ingest() を直接呼ぶ。
    失敗時は sys.executable でサブプロセス実行（仮想環境の Python を確実に使うため）。
    """
    try:
        import ingest  # type: ignore
        if hasattr(ingest, "run_ingest"):
            ingest.run_ingest()
            return True
    except Exception:
        pass

    try:
        # fix: "python" ではなく sys.executable を使うことで
        #      仮想環境の Python インタプリタを確実に使用する
        subprocess.check_call([sys.executable, "ingest.py"])
        return True
    except Exception as e:
        st.error(f"インデックス再構築に失敗しました: {e}")
        return False


def load_llm(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY が設定されていません（.env を確認してください）")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@st.cache_resource
def load_vectorstore_cached() -> Optional[FAISS]:
    # ※ここに widget を置かない（CachedWidgetWarning 回避）
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        vs = FAISS.load_local(
            INDEX_DIR,
            embeddings=embeddings,
            # NOTE: FAISS インデックスは Pickle 形式で保存されます。
            # 信頼できるローカル環境で生成したインデックスのみ読み込んでください。
            # 外部から受け取ったインデックスファイルには使用しないでください。
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception:
        return None


def clear_vectorstore_cache():
    load_vectorstore_cached.clear()


def get_sources_from_docs(docs) -> List[str]:
    return [d.metadata.get("source", "unknown") for d in docs]


def normalize_query_tokens(query: str, max_tokens: int = 8) -> List[str]:
    """
    ハイライト用トークン抽出。
    - 半角スペース区切りがあればそれを優先（英語・混在クエリ向け）
    - スペースなしの場合はクエリ全体を1トークンとして扱う（日本語向け）
    ※ 形態素解析は未実装のため、日本語では部分一致ハイライトになります。
    """
    q = (query or "").strip()
    if not q:
        return []

    parts = [p.strip() for p in re.split(r"\s+", q) if p.strip()]
    tokens = parts if len(parts) >= 2 else [q]

    # 短すぎるトークンは除外（記号や1文字など）
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens[:max_tokens]


def highlight_to_html(text: str, query: str) -> str:
    """
    根拠チャンクの表示を HTML でハイライト。
    - 文字列は escape し、マッチ部分のみ <mark> を付ける
    - 長いトークン優先で置換（部分一致の暴発を減らす）
    """
    safe = html.escape(text).replace("\n", "<br>")
    tokens = normalize_query_tokens(query)
    if not tokens:
        return safe

    tokens = sorted(tokens, key=len, reverse=True)
    for t in tokens:
        pattern = re.compile(re.escape(html.escape(t)), flags=re.IGNORECASE)
        safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return safe


def init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []


def add_history_item(item: Dict[str, Any]):
    st.session_state["history"].insert(0, item)  # 新しいものを先頭に


# =========================
# Sidebar（設定 + 履歴 + PDF管理）
# =========================
ensure_dirs()
init_history()

st.sidebar.header("⚙️ 設定")

model_name = st.sidebar.selectbox(
    "使用するLLM（OpenRouter）",
    [
        "deepseek/deepseek-v3.2",
        "x-ai/grok-4.1-fast",
        "openai/gpt-4o-mini",
    ],
    index=2,
)

temperature = st.sidebar.slider(
    "temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
)

highlight_enabled = st.sidebar.toggle(
    "根拠チャンクをハイライト表示",
    value=True,
)

if st.sidebar.button("🧹 キャッシュクリア（FAISS再読込）"):
    clear_vectorstore_cache()
    st.sidebar.success("キャッシュをクリアしました。必要なら再度実行してください。")

st.sidebar.divider()

# ---- 履歴（メタ情報＋参照PDF） ----
with st.sidebar.expander("🕘 質問履歴（メタ情報＋参照PDF）", expanded=False):
    if st.session_state["history"]:
        if st.button("履歴を全消去"):
            st.session_state["history"] = []
            st.success("履歴を消去しました。")
        for i, h in enumerate(st.session_state["history"][:30], 1):
            title = f"{i}. {h['time']} | {h['mode']} | {h['model']}"
            st.markdown(f"**{title}**")
            st.caption(f"Q: {h['question']}")
            if h.get("selected_pdf"):
                st.caption(f"PDF: {h['selected_pdf']}")
            if h.get("sources"):
                st.caption("参照: " + ", ".join(h["sources"]))
            st.markdown("---")
    else:
        st.caption("まだ履歴がありません。")

st.sidebar.divider()

# ---- PDF管理（アップロード/削除/再構築） ----
st.sidebar.header("📁 PDF管理")

uploaded_files = st.sidebar.file_uploader(
    "PDFを追加（複数可）",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.sidebar.caption("※「保存」ボタンを押すまで docs/ には書き込みません。")

if st.sidebar.button("💾 アップロードPDFを保存"):
    if not uploaded_files:
        st.sidebar.warning("アップロードされたPDFがありません。")
    else:
        saved = 0
        for f in uploaded_files:
            save_path = os.path.join(DOCS_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            saved += 1
        st.sidebar.success(f"{saved} 件保存しました（{DOCS_DIR}/）")

pdf_files = list_pdfs()
to_delete = st.sidebar.multiselect(
    "削除するPDF（複数可）",
    options=pdf_files,
)

if st.sidebar.button("🗑 選択したPDFを削除"):
    if not to_delete:
        st.sidebar.warning("削除対象が選択されていません。")
    else:
        deleted = 0
        for name in to_delete:
            path = os.path.join(DOCS_DIR, name)
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
        st.sidebar.success(f"{deleted} 件削除しました。")

st.sidebar.divider()

if st.sidebar.button("🔁 インデックス再構築（ingest）"):
    with st.spinner("インデックスを再構築しています…"):
        ok = run_ingest()
        clear_vectorstore_cache()
    if ok:
        st.sidebar.success("インデックス再構築が完了しました。")

# =========================
# インデックス状態表示
# =========================
vectorstore = load_vectorstore_cached()

col_a, col_b = st.columns([1, 2])
with col_a:
    st.subheader("📌 状態")
    st.write(f"- docs/ 内PDF: **{len(list_pdfs())}** 件")
    st.write(f"- インデックス: **{'あり' if vectorstore else 'なし'}**")
with col_b:
    st.subheader("🗂 docs/ のPDF一覧")
    if pdf_files:
        st.write(", ".join(pdf_files))
    else:
        st.info("docs/ にPDFがありません。")

st.divider()

# =========================
# 検索UI
# =========================
mode = st.radio(
    "検索モード",
    ["単一PDF", "全PDF横断"],
    horizontal=True,
)

selected_pdf = None
if mode == "単一PDF":
    if not pdf_files:
        st.warning("docs/ にPDFがありません。PDFを追加してください。")
    else:
        selected_pdf = st.selectbox("対象PDFを選択", pdf_files)

question = st.text_area(
    "質問（単一PDFモードでは空欄＝要約、全PDF横断モードでは空欄不可）",
    value="",
    height=120,
)

run = st.button("実行")

# =========================
# 実行
# =========================
if run:
    if not vectorstore:
        st.error("インデックスがありません。サイドバーの「インデックス再構築（ingest）」を実行してください。")
        st.stop()

    if mode == "単一PDF" and not selected_pdf:
        st.warning("単一PDFを選択してください。")
        st.stop()

    if mode == "全PDF横断" and not question.strip():
        st.warning("全PDF横断モードでは質問を入力してください（空欄要約は不可）。")
        st.stop()

    with st.spinner("検索・生成中…"):
        # 検索クエリ補完（単一PDFの空欄要約のみ対応）
        search_query = question.strip()
        is_summary = False
        if mode == "単一PDF" and not search_query:
            search_query = "この文書の全体概要 要点 まとめ"
            is_summary = True

        # 検索
        if mode == "単一PDF":
            docs = vectorstore.similarity_search(
                search_query,
                k=8,
                filter={"source": selected_pdf},
            )
        else:
            docs = vectorstore.similarity_search(search_query, k=12)

        if not docs:
            st.warning("該当する情報が見つかりませんでした（検索ヒット0件）。")
            st.stop()

        # 参照元（出典付き回答のために先に作る）
        sources = sorted(set(get_sources_from_docs(docs)))
        counts = Counter(get_sources_from_docs(docs))

        # コンテキスト作成
        context = "\n\n".join(d.page_content for d in docs)

        # 生成
        llm = load_llm(model_name=model_name, temperature=temperature)
        if mode == "単一PDF" and is_summary:
            messages = SUMMARY_PROMPT.format_messages(context=context)
            display_question = "(要約)"
        else:
            messages = QA_PROMPT.format_messages(context=context, question=question)
            display_question = question.strip()

        response = llm.invoke(messages)

        # 履歴保存（メタ情報＋参照PDF）
        add_history_item(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "question": display_question,
                "selected_pdf": selected_pdf if mode == "単一PDF" else None,
                "model": model_name,
                "temperature": temperature,
                "sources": sources,
            }
        )

    st.caption(f"Mode: {mode} / Model: {model_name} / temperature: {temperature}")
    st.subheader("📝 回答")
    st.write(response.content)

    # 出典付き回答：参照元PDFを回答直下に表示
    st.markdown("**参照元（検索ヒット由来）:** " + (", ".join(sources) if sources else "なし"))

    # 全PDF横断：PDF寄与度バー（チャンク数）
    if mode == "全PDF横断":
        st.subheader("📊 PDF寄与度（参照チャンク数）")
        chart_data = {k: v for k, v in counts.most_common()}
        st.bar_chart(chart_data)

    # 参照元の詳細表示（全PDF横断はPDF別にまとめる）
    if mode == "全PDF横断":
        st.subheader("📚 参照されたPDF（詳細）")
        st.caption("※検索でヒットしたチャンクの出典です（回答が参照した可能性が高いPDF）")
        for name, cnt in counts.most_common():
            st.markdown(f"- **{name}**（参照チャンク: {cnt}）")

        st.subheader("🔎 PDF別の参照チャンク（根拠）")
        for name, cnt in counts.most_common():
            with st.expander(f"{name}（{cnt} chunks）"):
                for i, d in enumerate([x for x in docs if x.metadata.get("source") == name], 1):
                    st.markdown(f"**Chunk {i}**")
                    if highlight_enabled:
                        st.markdown(
                            highlight_to_html(d.page_content, question),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(d.page_content)
                    st.markdown("---")
    else:
        with st.expander("🔍 参照された文書チャンク（根拠）"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}**")
                if highlight_enabled and (question.strip() or search_query.strip()):
                    q_for_hl = question if question.strip() else search_query
                    st.markdown(
                        highlight_to_html(d.page_content, q_for_hl),
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(d.page_content)
                st.caption(f"source: {d.metadata.get('source')}")
                st.markdown("---")

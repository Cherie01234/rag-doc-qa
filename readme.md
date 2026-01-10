# 📄 OpenRouter対応 RAG ドキュメントQA（Streamlit）

PDFドキュメントを検索・要約し、**複数のLLM（DeepSeek / Grok / GPT）を切り替えて**質問応答できる  
**RAG（Retrieval Augmented Generation）アプリケーション**です。

OpenRouter を介して複数モデルを統一的に扱い、  
FAISS + Sentence-Transformers による高速なローカル検索を組み合わせています。

---

## ✨ 主な特徴

### 🔍 RAG（検索拡張生成）
- PDFをチャンク分割 → ベクトル化 → FAISS に格納  
- 質問に応じて関連チャンクを検索し、LLMに渡して回答を生成  

### 🤖 複数LLM切り替え（OpenRouter）
UIから以下のモデルを切り替え可能：

- **DeepSeek V3.2**
- **Grok 4.1 Fast**
- **GPT-4o mini**

### 📁 PDF管理
- Streamlit UI から PDF をアップロード
- 不要なPDFの削除
- ワンクリックでインデックス再構築（ingest）

### 🧭 検索モード
| モード | 内容 |
|------|------|
| 単一PDF | 選択したPDFだけを対象に質問・要約 |
| 全PDF横断 | 全てのPDFを横断して検索・質問 |

※ 全PDF横断モードでは質問入力必須（意味のない全文要約を防止）

### 📌 出典付き回答
- 回答の直下に **参照されたPDF名を表示**

### 📊 PDF寄与度（全PDF横断時）
- 質問に対してどのPDFがどれだけ使われたかをバーグラフで可視化

### 🖍 根拠チャンクのハイライト
- 質問に関連する語句を参照チャンク内で強調表示（ON / OFF 切替可）

### 🕘 質問履歴
- 実行日時
- モード
- 質問
- 選択PDF
- 使用モデル
- 参照PDF一覧

---

## 🛠 技術スタック

| 分類 | 使用技術 |
|------|---------|
| UI | Streamlit |
| LLM | OpenRouter（DeepSeek / Grok / GPT-4o mini） |
| ベクトルDB | FAISS |
| 埋め込み | Sentence-Transformers (all-MiniLM-L6-v2) |
| RAGフレームワーク | LangChain |
| PDF読み込み | PyPDFLoader |
| 設定管理 | python-dotenv |

---

## 🚀 起動方法

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

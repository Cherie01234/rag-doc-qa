# 📄 RAG ドキュメント QA アプリ

PDF・テキストファイルに対して自然言語で質問できる、RAG（Retrieval-Augmented Generation）ベースの QA アプリです。  
LangChain + FAISS + Streamlit で構成されており、OpenRouter 経由で複数の LLM を切り替えて利用できます。

---

![アプリ画面](images/app_screen.png)

---

## 📌 主な機能

| 機能 | 説明 |
|---|---|
| ドキュメント QA | アップロードした PDF / TXT に対して自然言語で質問できる |
| 要約モード | 単一 PDF を選択して質問を空欄にすると、自動で要約を生成 |
| 全 PDF 横断検索 | 複数 PDF をまたいで関連チャンクを検索・回答 |
| PDF 寄与度表示 | 全 PDF 横断時に、各 PDF の参照チャンク数をバーチャートで可視化 |
| 根拠チャンク表示 | 回答の根拠となったテキストチャンクをハイライト付きで表示（Explainable RAG） |
| マルチ LLM 切り替え | DeepSeek / Grok / GPT-4o mini を UI から切り替え可能 |
| PDF 管理 | UI 上から PDF のアップロード・削除・インデックス再構築が可能 |
| 質問履歴 | セッション内の質問履歴をサイドバーに保持（参照 PDF・モデル情報付き） |

---

## 🏗 システム構成

```
ユーザー質問
    ↓
[Streamlit UI] (app.py)
    ↓ 埋め込みベクトル検索
[FAISS インデックス] ← [ingest.py でインデックス構築]
    ↓ 関連チャンク取得
[OpenRouter API] → LLM による回答生成
    ↓
回答 + 根拠チャンク表示
```

**使用技術**

- **フロントエンド**: Streamlit
- **RAG フレームワーク**: LangChain
- **ベクトルストア**: FAISS（ローカル保存）
- **埋め込みモデル**: `sentence-transformers/all-MiniLM-L6-v2`（HuggingFace・ローカル実行）
- **LLM**: OpenRouter API（DeepSeek / Grok / GPT-4o mini）

---

## 📁 ファイル構成

```
rag-doc-qa/
├── app.py              # Streamlit UI・検索・回答生成のメインアプリ
├── ingest.py           # PDF/TXT の読み込み・チャンク分割・FAISS インデックス構築
├── query.py            # CLI からの動作確認用スクリプト（任意）
├── config.py           # 定数の一元管理（パス・モデル名・チャンクサイズ等）
├── requirements.txt    # 依存パッケージ一覧（pip freeze 出力）
├── .env                # API キー設定ファイル（Git 管理外）
├── .gitignore
├── docs/               # 質問対象の PDF / TXT を格納するフォルダ
├── faiss_index/        # FAISS インデックスの保存先（自動生成）
└── README.md
```

---

## ⚙️ セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/Cherie01234/rag-doc-qa.git
cd rag-doc-qa
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env` ファイルをプロジェクトルートに作成し、以下を記載してください。

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

OpenRouter の API キーは [https://openrouter.ai](https://openrouter.ai) から取得できます。

---

## 🚀 使い方

### アプリの起動

```bash
streamlit run app.py
```

起動後、ブラウザで `http://localhost:8501` が自動で開きます。

### 基本的な流れ

1. サイドバーの **「PDF管理」** から PDF をアップロードして保存
2. **「インデックス再構築（ingest）」** ボタンを押す
3. 検索モードを選択（単一 PDF / 全 PDF 横断）
4. 質問を入力して **「実行」** ボタンを押す
5. 回答と根拠チャンク（ハイライト付き）が表示される

### CLI での動作確認（任意）

```bash
# インデックス構築のみ実行
python ingest.py

# CLI から質問
python query.py
```

---

## 🔧 カスタマイズ

`config.py` を編集することで、主要パラメータを一元管理できます。

```python
DOCS_DIR = "docs"          # PDF 格納フォルダ
INDEX_DIR = "faiss_index"  # インデックス保存先
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800           # チャンクサイズ（日本語向けに調整済み）
CHUNK_OVERLAP = 100        # チャンク間のオーバーラップ
```

> **チューニングの目安**  
> - 日本語文書が多い場合: `CHUNK_SIZE = 800〜1000`  
> - 英語文書が多い場合: `CHUNK_SIZE = 500〜600`  
> - 文脈の連続性を重視する場合: `CHUNK_OVERLAP` を大きく（150〜200）

---

## ⚠️ セキュリティについて

FAISS インデックスは Python の Pickle 形式で保存されます。  
アプリ起動時に `allow_dangerous_deserialization=True` を使用してインデックスを読み込んでいます。

**信頼できるローカル環境で生成したインデックスのみ使用してください。**  
外部から受け取ったインデックスファイル（`faiss_index/` フォルダ）は読み込まないでください。

---

## 📝 ハイライト機能について

根拠チャンクのハイライト表示は、クエリのキーワードをテキスト内で部分一致検索して色付けします。

- **英語・スペース区切りのクエリ**: 単語単位でハイライト
- **日本語クエリ**: クエリ文字列全体での部分一致ハイライト（形態素解析は未実装）

---

## 🛠 動作環境

- Python 3.10 以上
- Windows / Mac / Linux
- インターネット接続（OpenRouter API の利用・埋め込みモデルの初回ダウンロードに必要）

---

## 📜 ライセンス

MIT License

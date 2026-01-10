📄 OpenRouter対応 RAG ドキュメントQA（Streamlit）



PDFドキュメントを検索・要約し、複数のLLM（DeepSeek / Grok / GPT）を切り替えて質問応答できる

RAG（Retrieval Augmented Generation）アプリケーションです。



OpenRouter を介して複数モデルを統一的に扱い、

FAISS + Sentence-Transformers による高速なローカル検索を組み合わせています。



✨ 主な特徴

🔍 RAG（検索拡張生成）



PDFをチャンク分割 → ベクトル化 → FAISS に格納



質問に応じて関連チャンクを検索し、LLMに渡して回答を生成



🤖 複数LLM切り替え（OpenRouter）



以下のモデルをUIから切り替え可能：



DeepSeek V3.2



Grok 4.1 Fast



GPT-4o mini



用途（要約 / 精度 / 速度）に応じてモデルを選択できます。



📁 PDF管理



Streamlit UI から PDF をアップロード



不要なPDFの削除



ワンクリックでインデックス再構築（ingest）



🧭 検索モード

モード	内容

単一PDF	選択したPDFだけを対象に質問・要約

全PDF横断	全てのPDFを横断して検索・質問



※ 全PDF横断モードでは質問入力必須（意味のない全文要約を防止）



📌 出典付き回答（Explainable RAG）



回答の直下に 参照されたPDF名を表示



どの資料を根拠にしたかが一目で分かる



📊 PDF寄与度（全PDF横断時）



質問に対して

どのPDFがどれだけ使われたか をバーグラフで可視化。



→ 「この回答はどの資料に依存しているか」を定量的に確認可能



🖍 根拠チャンクのハイライト



質問に関連する語句を

参照チャンク内で強調表示（ON / OFF 切替可）



→ 検索と生成のつながりを視覚的に理解できる



🕘 質問履歴（メタデータ）



サイドバーに以下を保存：



実行日時



モード（単一PDF / 全PDF）



質問



選択PDF



使用モデル



参照されたPDF一覧



→ 再現性・検証性を意識した設計



🛠 技術スタック

分類	使用技術

UI	Streamlit

LLM	OpenRouter（DeepSeek / Grok / GPT-4o mini）

ベクトルDB	FAISS

埋め込み	Sentence-Transformers (all-MiniLM-L6-v2)

RAGフレームワーク	LangChain

PDF読み込み	PyPDFLoader

設定管理	python-dotenv

🧩 アーキテクチャ概要

PDF → チャンク分割 → Embedding → FAISS

&nbsp;                              ↓

&nbsp;                        質問ベクトル

&nbsp;                              ↓

&nbsp;                   関連チャンク検索

&nbsp;                              ↓

&nbsp;               LLM（OpenRouter）で生成

&nbsp;                              ↓

&nbsp;             回答 + 出典PDF + 根拠表示



🚀 起動方法

1\. 仮想環境と依存関係

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt



2\. OpenRouter APIキー



.env を作成：



OPENROUTER\_API\_KEY=sk-xxxxxx



3\. インデックス作成

python ingest.py



4\. Streamlit 起動

streamlit run app.py





または

run\_app.bat をダブルクリック。



📂 フォルダ構成

rag-doc-qa/

├── app.py        # Streamlit UI

├── ingest.py     # PDF → FAISS 変換

├── query.py      # CLIテスト用（任意）

├── docs/         # PDF配置

├── faiss\_index/  # ベクトルDB

├── run\_app.bat

├── .env

└── requirements.txt



🧠 設計のポイント



埋め込みモデルとLLMを分離



Embeddingは高速で無料なローカルモデル



生成はOpenRouter経由の高性能LLM



Explainable RAG



回答だけでなく「どのPDFを使ったか」を明示



運用を想定したUI



PDFの差し替え



再インデックス



モデル切り替え



履歴管理


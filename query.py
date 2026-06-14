# query.py  ― CLI テスト用（任意）
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # fix: app.py と統一
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import INDEX_DIR, EMBEDDING_MODEL

load_dotenv()


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
        # NOTE: FAISS インデックスは Pickle 形式で保存されます。
        # 信頼できるローカル環境で生成したインデックスのみ読み込んでください。
        # 外部から受け取ったインデックスファイルには使用しないでください。
    )

    query = input("質問を入力してください: ")
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = f"""
以下は参考文書です。
----------------
{context}
----------------
この情報をもとに、次の質問に日本語で簡潔に答えてください。
質問: {query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("\n=== 回答 ===")
    print(response.content)


if __name__ == "__main__":
    main()

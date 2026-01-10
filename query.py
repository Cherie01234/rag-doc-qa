from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "faiss_index"

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    query = input("質問を入力してください: ")

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
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

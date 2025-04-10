import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

def load_nlplog_vectorstore(embedding_model, path="data/synthetic_nlplog_dataset_extended.json"):
    with open(path, "r") as f:
        raw_data = json.load(f)

    docs = []
    for item in raw_data:
        question = item.get("instruction", "").strip()
        answer = item.get("output", "").strip()
        text = f"Q: {question}\nA: {answer}"
        docs.append(Document(page_content=text))

    return FAISS.from_documents(docs, embedding_model)
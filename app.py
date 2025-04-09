from flask import Flask, request, jsonify
from graylog_loader import get_recent_logs
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

def create_vector_store(logs):
    global vector_store
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([log["message"] for log in logs])
    vector_store = FAISS.from_documents(docs, embedding_model)

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    logs = get_recent_logs(minutes=5)
    if not logs:
        return jsonify({"answer": "No logs found in last 5 minutes."})

    create_vector_store(logs)

    llm = OllamaLLM(model="deepseek-r1:7b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    user_q = request.json.get("question")
    response = qa.invoke(user_q)

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)

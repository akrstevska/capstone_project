# first version without similarity search

# from flask import Flask, request, jsonify
# from graylog_loader import get_recent_logs
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import CharacterTextSplitter

# app = Flask(__name__)

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = None

# def create_vector_store(logs):
#     global vector_store
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.create_documents([log["message"] for log in logs])
#     vector_store = FAISS.from_documents(docs, embedding_model)

# @app.route("/ask", methods=["POST"])
# def ask():
#     global vector_store
#     logs = get_recent_logs(minutes=5)
#     if not logs:
#         return jsonify({"answer": "No logs found in last 5 minutes."})

#     create_vector_store(logs)

#     llm = OllamaLLM(model="deepseek-r1:7b")
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

#     user_q = request.json.get("question")
#     response = qa.invoke(user_q)

#     return jsonify({"answer": response})

# if __name__ == "__main__":
#     app.run(debug=True)


# with similarity search from synthetic_nlplog_dataset_extended.json

# tried it with NLPLog.json from SuperLog but its too huge and the logs aren't really in the domain i'm looking for so i created a synthetic one with 10-20 Q&A

from flask import Flask, request, jsonify
from graylog_loader import get_recent_logs
from nlplog_utils import load_nlplog_vectorstore

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
nlplog_store = load_nlplog_vectorstore(embedding_model)
vector_store = None

def create_vector_store(logs):
    global vector_store
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([log["message"] for log in logs])
    vector_store = FAISS.from_documents(docs, embedding_model)

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    logs = get_recent_logs(minutes=10)
    if not logs:
        return jsonify({"answer": "No logs found in last 5 minutes."})
    # return jsonify({"log_count": len(logs)})
    create_vector_store(logs)

    user_q = request.json.get("question")

    similar_docs = nlplog_store.similarity_search(user_q, k=3)
    context_examples = "\n\n".join([doc.page_content for doc in similar_docs])

    enriched_prompt = f"""Here are some example questions and answers related to log analysis:\n\n{context_examples}\n\nNow answer this user question:\n{user_q}"""

    llm = OllamaLLM(model="deepseek-r1:7b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    response = qa.invoke(enriched_prompt)

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
# first version without similarity search

# from flask import Flask, request, jsonify
# from graylog_loader import get_recent_logs
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import CharacterTextSplitter

# app = Flask(__name__)

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = None

# def create_vector_store(logs):
#     global vector_store
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.create_documents([log["message"] for log in logs])
#     vector_store = FAISS.from_documents(docs, embedding_model)

# @app.route("/ask", methods=["POST"])
# def ask():
#     global vector_store
#     logs = get_recent_logs(minutes=5)
#     if not logs:
#         return jsonify({"answer": "No logs found in last 5 minutes."})

#     create_vector_store(logs)

#     llm = OllamaLLM(model="deepseek-r1:7b")
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

#     user_q = request.json.get("question")
#     response = qa.invoke(user_q)

#     return jsonify({"answer": response})

# if __name__ == "__main__":
#     app.run(debug=True)


# with similarity search from synthetic_nlplog_dataset_extended.json

# tried it with NLPLog.json from SuperLog but its too huge and the logs aren't really in the domain i'm looking for so i created a synthetic one with 10-20 Q&A

from flask import Flask, request, jsonify
from graylog_loader import get_recent_logs
from nlplog_utils import load_nlplog_vectorstore

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
nlplog_store = load_nlplog_vectorstore(embedding_model)
vector_store = None

def create_vector_store(logs):
    global vector_store
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([log["message"] for log in logs])
    vector_store = FAISS.from_documents(docs, embedding_model)

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    logs = get_recent_logs(minutes=10)
    if not logs:
        return jsonify({"answer": "No logs found in last 5 minutes."})
    # return jsonify({"log_count": len(logs)})
    create_vector_store(logs)

    user_q = request.json.get("question")

    similar_docs = nlplog_store.similarity_search(user_q, k=3)
    context_examples = "\n\n".join([doc.page_content for doc in similar_docs])

    enriched_prompt = f"""Here are some example questions and answers related to log analysis:\n\n{context_examples}\n\nNow answer this user question:\n{user_q}"""

    llm = OllamaLLM(model="deepseek-r1:7b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    response = qa.invoke(enriched_prompt)

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)

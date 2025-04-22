from flask import Flask, request, jsonify
from agent_setup import agent 

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_q = request.json.get("question")
        if not user_q:
            return jsonify({"error": "No question provided"}), 400
        
        response = agent.invoke(user_q)  
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
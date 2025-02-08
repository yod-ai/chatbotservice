from flask import Flask, request, jsonify
from v4 import PetAssistant  # ensure chatbot.py is in the same folder
from uuid import uuid4

app = Flask(__name__)

# Instantiate the chatbot service once; it manages its own state.
assistant = PetAssistant()


@app.route("/chat", methods=["POST"])
def chat():
    """
    Expects JSON with keys:
      - user_id: a unique user identifier
      - session_id: the current chat session identifier
      - query: the user's input query
    Returns the chatbot's response.
    """
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    session_id = data.get("session_id")
    query = data.get("query")

    if not user_id or not session_id or not query:
        return jsonify({"error": "Missing required fields: user_id, session_id, and query"}), 400

    # Prepare the state dictionary for the chatbot workflow.
    state = {
        "messages": [],
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "query_type": "",
        "retrieved_docs": [],
        "current_response": "",
        "requires_human": False,
        "reflection": None,
        "confidence_score": 0.0,
        "has_pet": False,
        "pet_info": None
    }
    try:
        final_state = assistant.workflow.invoke(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "response": final_state["current_response"],
        "confidence_score": final_state["confidence_score"],
        "reflection": final_state["reflection"]
    })


if __name__ == "__main__":
    app.run(port=3010, debug=True)
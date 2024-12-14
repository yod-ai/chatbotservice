from flask import Flask, request, jsonify
from chatbot_module import chatbot  # Import your chatbot definition

app = Flask(__name__)

def invoke_chatbot(inputs):
    return chatbot.invoke(inputs)

@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    try:
        data = request.get_json(force=True)  # Parse JSON
        if not data or 'question' not in data:
            return jsonify({"error": "Invalid input. Please provide a 'question' field."}), 400
        response = invoke_chatbot(data)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

@app.route('/')
def index():
    return "Welcome to the Chatbot API! Use the '/chatbot' endpoint to interact with the bot.", 200

if __name__ == "__main__":
    app.run(port=5001, debug=True)
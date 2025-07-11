from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import json
import os
import random
from NLP import process_query  # Assuming you have an NLP.py with process_query function

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load greeting.json
greeting_file_path = os.path.join(os.getcwd(), 'Dataset', 'greeting.json')
greeting_data = {}
try:
    with open(greeting_file_path, 'r', encoding='utf-8') as file:
        greeting_data = json.load(file)
except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
    logging.error(f"Error loading greeting.json: {e}")

# In-memory chat history
chat_history = []

def get_random_response(responses):
    """Selects a random response from a list."""
    if responses and isinstance(responses, list):
        return random.choice(responses)
    elif responses:
        return responses
    return None

@app.route('/')
def hello():
    return "Education Chatbot Backend is running!"

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data['query']
        logging.info(f"Received query: {query}")

        response_content = None
        suggestions = []
        
        # Normalize query for easier matching
        normalized_query = query.lower().strip()

        # Check for greeting - improved pattern matching
        greeting_found = False
        for keyword, details in greeting_data.get('greetings', {}).items():
            if keyword in normalized_query or normalized_query in keyword:
                response_content = get_random_response(details.get('response'))
                suggestions = details.get('suggestions', [])
                greeting_found = True
                break
                
        # For very short greetings (hi, hey, etc), do a more lenient check
        if not greeting_found and len(normalized_query.split()) == 1:
            for keyword in ['hi', 'hey', 'hello', 'sup', 'yo', 'greetings']:
                if normalized_query == keyword:
                    for greet_key, details in greeting_data.get('greetings', {}).items():
                        if greet_key in ['hi', 'hello', 'hey']:
                            response_content = get_random_response(details.get('response'))
                            suggestions = details.get('suggestions', [])
                            greeting_found = True
                            break
                    if greeting_found:
                        break

        # Check for ending message - improved pattern matching
        if not response_content:
            for keyword, details in greeting_data.get('ending_messages', {}).items():
                if keyword in normalized_query or normalized_query in ['bye', 'goodbye', 'farewell', 'see you', 'thanks', 'thank you', 'have a nice day']:
                    response_content = get_random_response(details.get('response'))
                    suggestions = details.get('suggestions', [])
                    break

        # If no specific greeting or ending is found, process as a general query
        if not response_content:
            # process_query now returns both response and suggestions
            processed_output = process_query(query)
            if isinstance(processed_output, tuple) and len(processed_output) == 2:
                response_content, suggestions = processed_output
            else:
                response_content = processed_output
                suggestions = []

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response_content})

        # If no suggestions were provided, add default ones
        if not suggestions:
            suggestions = [
                "Tell me about another career",
                "What do you know?",
                "Let me teach you something"
            ]

        return jsonify({"response": response_content, "suggestions": suggestions, "history": chat_history}), 200

    except Exception as e:
        logging.error(f"Error in /chat route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": chat_history}), 200

@app.route('/clear_history', methods=['POST'])
def clear_history():
    chat_history.clear()
    return jsonify({"message": "Chat history cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
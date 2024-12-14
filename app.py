from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    # Get the user message from the form data
    msg = request.form["msg"]
    response_text = get_chat_response(msg)
    return jsonify(response_text)

def get_chat_response(text):
    # Initialize chat history as an empty tensor (or None if conversation hasn't started)
    chat_history_ids = None

    # Process input for one conversational step
    # Encode user input and add end-of-sequence token
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    # Append new user input to chat history if available, otherwise start with user input
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate response, keeping history limited to 1000 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return response text from the model
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(debug=True)
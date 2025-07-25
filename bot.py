from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import random
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# === Load and Validate intent.json ===
if not os.path.exists("intent.json") or os.path.getsize("intent.json") == 0:
    raise ValueError("intent.json file is missing or empty. Please provide valid JSON content.")

with open("intent.json", "r", encoding="utf-8") as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError("intent.json contains invalid JSON.")

# === Prepare Data ===
corpus = []
tags = []
responses = {}

for item in data:
    intent = item.get("intent")
    patterns = item.get("patterns", [])
    intent_responses = item.get("responses", [])

    for pattern in patterns:
        corpus.append(pattern)
        tags.append(intent)

    responses[intent] = intent_responses

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

stop_words = set(stopwords.words("english"))

# === Helper Functions ===
def correct_spelling(text):
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text

def preprocess_input(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def load_chat_history():
    return session.get('chat_history', [])

def save_chat_history(chat_history):
    session['chat_history'] = chat_history

def get_bot_response(user_input):
    if not corpus:
        return "Bot training data is missing or invalid."

    corrected_input = correct_spelling(user_input)
    processed_input = preprocess_input(corrected_input)
    user_vec = vectorizer.transform([processed_input])
    sim_scores = cosine_similarity(user_vec, X)
    best_match_index = sim_scores.argmax()
    confidence = sim_scores[0, best_match_index]

    if confidence > 0.2:
        best_intent = tags[best_match_index]
        return random.choice(responses.get(best_intent, ["I'm not sure how to respond to that."]))
    else:
        return random.choice(responses.get("fallback", [
            "I'm sorry, I didn't understand that. Could you please rephrase?",
            "Hmm, I couldn't catch that. Can you try saying it differently?",
            "Oops! That doesn't seem to match anything I know. Try asking something else.",
            "Sorry, I didn’t get that. You can ask me about our services or features.",
            "I’m still learning! You can try asking in a different way."
        ]))

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def home():
    chat_history = load_chat_history()
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            response = get_bot_response(query)
            chat_history.append({'role': 'user', 'text': query})
            chat_history.append({'role': 'bot', 'text': response})
            save_chat_history(chat_history)
    return render_template('index.html', chat_history=chat_history)

@app.route("/chat", methods=["POST"])
def api_chat_post():
    if not request.is_json:
        return jsonify({"response": random.choice(responses.get("fallback", ["Invalid request format."]))}), 415
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a valid message."})
    response = get_bot_response(user_input)
    return jsonify({"response": response})

@app.route("/chat", methods=["GET"])
def api_chat_get():
    user_input = request.args.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please provide a message in the query parameter."})
    response = get_bot_response(user_input)
    return jsonify({"response": response})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('chat_history', None)
    return redirect(url_for('home'))

# === Run the App ===
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

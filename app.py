from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.datasets import imdb      # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import os

app = Flask(__name__)

# Load the model
model = load_model('sentiment_model.keras')

# Load word index
word_index = imdb.get_word_index()
top_words = 10000
max_review_length = 300

def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # <START>
    for word in words:
        if word in word_index and word_index[word] < top_words:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # <UNK>
    return pad_sequences([encoded], maxlen=max_review_length)

def predict_sentiment(text):
    encoded = encode_review(text)
    prediction = model.predict(encoded, verbose=0)[0][0]
    if prediction < 0.4:
        return "Negative"
    elif prediction > 0.6:
        return "Positive"
    else:
        return "Neutral"

@app.route('/')
def home():
    return render_template('index.html', title="Moodie")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

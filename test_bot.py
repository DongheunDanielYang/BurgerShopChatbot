# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 15:37:17 2025

@author: DongheunYang
"""

import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved components
with open("dongheun_intents.json", "r") as file:
    intents = json.load(file)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = load_model("chatbot_model.h5")

# Start chatbot
print("CoffeeBot is ready! Type 'bye' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Goodbye! Have a nice day ☕")
        break

    # Convert input to sequence
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=40, padding='post')

    # Predict intent
    prediction = model.predict(padded)
    tag_index = np.argmax(prediction)
    predicted_tag = label_encoder.inverse_transform([tag_index])[0]

    # Find response
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            print("Bot:", response)
            break

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:34:11 2025

@author: DongheunYang
"""

import json

with open("dongheun_intents.json") as file:
    data = json.load(file)
    
all_patterns = []
all_tags = []
all_responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        tag = intent['tag']
        all_tags.append(tag)
        all_patterns.append(pattern)
    all_responses[tag] = intent['responses']    

        
        
print(f"total num of patterns: {len(all_patterns)}")
print(f"example pattern: {all_patterns[:10]}")
print(f"example tag: {all_tags[:5]}")
print(f"example response: {all_responses[tag][:5]}")
print(f"all responses: {all_responses}")

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_tags)

tokenizer = Tokenizer(num_words=900)
tokenizer.fit_on_texts(all_patterns)
sequences = tokenizer.texts_to_sequences(all_patterns)
word_index = tokenizer.word_index
print(f"word of index {word_index}")
 
max_len = 40
padded_sequences = pad_sequences(sequences, maxlen=max_len)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input


vocab_size = 900
embedding_dim = 20
input_length = 40
num_classes = len(label_encoder.classes_)

# Model Composition
model = Sequential([
    Input(shape=(40,)),
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
    GlobalAveragePooling1D(),
    
    Dense(16, activation='relu'),
    Dense(10, activation='sigmoid'),
    Dense(num_classes, activation='softmax')
])
 


# Compiling
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Model Composition Output
model.summary()

model.fit(padded_sequences, encoded_labels)

# Saving the model and Preprocessing
model.save("chatbot_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model, tokenizer, and label encoder saved successfully.")


























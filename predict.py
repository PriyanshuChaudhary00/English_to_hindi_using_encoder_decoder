import torch
import torch.nn as nn
import torch.optim as optim 
import os
import json
from model import getModel
from dataset import word_to_number, num_to_word

vocab_path = 'checkpoint/vocab.json'
if not os.path.exists(vocab_path):
    raise FileNotFoundError(
        f"'{vocab_path}' not found! Please run 'python train.py' first to train the model and save the corresponding vocabulary."
    )

with open(vocab_path, "r", encoding="utf-8") as f:
    vocabs = json.load(f)

vocab_english = vocabs["vocab_english"]
vocab_hindi = vocabs["vocab_hindi"]

idx_to_hindi = {int(v): k for k, v in vocab_hindi.items()}
model = getModel(vocab_english, vocab_hindi)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)


model.load_state_dict(torch.load('checkpoint/model_epoch10.pth', map_location=device, weights_only=True))
print("\n\nSome Examples \n\n")
for i in range(5):
    word = input("enter any thing in english :\t")
    word = word_to_number(word.lower().split() , vocab_english)
    # print(word)
    word = torch.tensor(word , dtype=torch.long).unsqueeze(0).to(device)
    model.eval()

    predict = model.predict(word)
    hindi_output = num_to_word(predict , idx_to_hindi)
    print("Hindi Output \t\t   : \t" , hindi_output)

import random
import json
import torch
import numpy as np

import google.generativeai as genai 

from preprocessing import bag_of_words, tokenize, stem
from train import NeuralNet 

# WIKI CONTEXT
wiki_context = """
Kamu adalah asisten server Hytale bernama WikiBot.
Jawab pertanyaan user berdasarkan data berikut:

DATA SERVER:
- Nama Server: Tales of Nusantara.
- Mata uang: HytCoin (didapat dari membunuh monster).
- Admin: iKagu, Dackin, Keinsteyn.
- Cara Donasi: Transfer ke tako.id/talesofnusantara.
- Rules: Dilarang toxic, dilarang cheat X-Ray.

Jika user bertanya di luar topik game/server, tolak dengan sopan.
"""

# --- PERBAIKAN 2: Setup Model Gemini (Cara Standar) ---
API_KEY = "AIzaSyDh4fHIO2fbiYunYQV4kkO0dEx3xW1mpg4"  # API KEY KAMU AMAN
genai.configure(api_key=API_KEY)

# Inisialisasi Model
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

def tanya_gemini(user_input):
    try:
        # Gabungkan konteks + input user
        prompt_lengkap = wiki_context + "\nUser: " + user_input + "\nWikiBot:"
        
        # --- PERBAIKAN 3: Panggil generate dari objek model ---
        response = model_gemini.generate_content(prompt_lengkap)
        return response.text
    except Exception as e:
        print(f"\n[ERROR GEMINI]: {e}")
        return "Maaf, terjadi kesalahan saat menghubungi model Gemini."

# MODEL LOKAL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. LOAD DATA DAN MODEL
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# 2. FUNGSI CHAT UTAMA
bot_name = "NusantaraBot"
print("Bot sudah siap! Ketik 'quit' untuk keluar.")
print("---------------------------------------------")

while True:
    sentence = input("Kamu: ")
    if sentence == "quit":
        break

    # --- PROSES 1: Tokenisasi ---
    tokenize_sentence = tokenize(sentence)
    X = bag_of_words(tokenize_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # --- PROSES 2: Prediksi Model Lokal ---
    output = model(X)    
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Hitung probabilitas
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # --- PROSES 3: Cek Keyakinan (Threshold) ---
    if prob.item() > 1:
        # Jika yakin > 90%, pakai jawaban lokal
        print(f"[DEBUG: Model Lokal Yakin ({prob.item():.2f})]")
        for intent in intents['intents']:
            if intent['tag'] == tag:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        # Jika bingung, lempar ke Gemini
        print(f"[DEBUG: Beralih ke Gemini, Confidence cuma: {prob.item():.2f}]")
        print(f"{bot_name} (Thinking...): ", end="", flush=True)
        jawaban_gemini = tanya_gemini(sentence)
        print(jawaban_gemini)
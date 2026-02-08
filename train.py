import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocessing import bag_of_words, tokenize, stem

# --- 1. KONFIGURASI DATASET (ETL: Extract, Transform, Load) ---
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop data json untuk memisahkan kata
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize (memecah kalimat)
        w = tokenize(pattern)
        all_words.extend(w)
        # Simpan pasangan (pola, tag)
        xy.append((w, tag))

# Stemming & Cleaning (Pakai Sastrawi dari preprocessing.py)
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Himpunan Unik (Sorted Set)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"Jumlah pola (kalimat contoh): {len(xy)}")
print(f"Jumlah tag (kategori): {len(tags)}")
print(f"Jumlah vocabulary unik: {len(all_words)}")

# Membuat Training Data (X dan y)
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: Bag of words (Vektor Input)
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    # y: Label (Index dari tag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# --- 2. CLASS DATASET (Standar PyTorch) ---
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Fungsi untuk mengambil data berdasarkan index (matrix row)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Fungsi untuk cek panjang data
    def __len__(self):
        return self.n_samples

# --- 3. ARSITEKTUR MODEL (Neural Network) ---
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Layer 1: Input -> Hidden
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU() # Aktivasi (agar tidak linear murni)
        
        # Layer 2: Hidden -> Hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)
        
        # Layer 3: Hidden -> Output
        self.l3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # Tidak perlu Softmax di sini karena nanti pakai CrossEntropyLoss
        return out

# --- 4. HYPERPARAMETERS (Settingan Training) ---
num_epochs = 1000       # Berapa kali bot membaca ulang seluruh data
batch_size = 8          # Sekali baca langsung 8 kalimat
learning_rate = 0.001   # Kecepatan update otak (jangan terlalu ngebut)
input_size = len(X_train[0]) # Panjang vektor vocabulary
hidden_size = 8         # Jumlah neuron di otak tengah
output_size = len(tags) # Jumlah kategori jawaban

# Setup Pipeline
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- 5. TRAINING LOOP (Proses Belajar) ---
if __name__ == '__main__':
    print("Mulai training model...")
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass (Maju)
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward and optimize (Mundur memperbaiki kesalahan)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Training selesai. Loss akhir: {loss.item():.4f}')

    # --- 6. SIMPAN DATA ---
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'Model berhasil disimpan ke {FILE}')
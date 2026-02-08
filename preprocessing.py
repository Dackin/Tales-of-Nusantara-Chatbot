import nltk
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab') # Versi baru kadang butuh ini juga
# Inisialisasi pemotong kata


# --- FUNGSI 1: Tokenization (Memecah Kalimat) ---
def tokenize(sentence):
    """
    Input: "Gimana cara join?"
    Output: ["Gimana", "cara", "join", "?"]
    """
    return nltk.word_tokenize(sentence)

# --- FUNGSI 2: Stemming (Mencari Kata Dasar) ---
def stem(word):
    """
    Mengubah kata ke bentuk dasarnya agar AI tidak bingung.
    Input: "makan", "makanan", "dimakan" -> Output: "makan"
    (Note: NLTK defaultnya bagus di Inggris, untuk Indo agak kaku tapi cukup untuk belajar)
    """
    return stemmer.stem(word.lower())

# --- FUNGSI 3: Bag of Words (Konsep Matriks/Vektor) ---
def bag_of_words(tokenized_sentence, all_words):
    """
    Mengubah kalimat menjadi array angka (biner).
    Konsep Matematika Diskrit: Himpunan Bagian & Vektor
    
    Jika all_words = ["halo", "cara", "join", "server"]
    Input kalimat = ["cara", "join"]
    
    Maka Output Vektor = [0, 1, 1, 0]
    (1 artinya kata itu muncul, 0 artinya tidak)
    """
    # Stemming dulu inputnya
    sentence_words = [stem(w) for w in tokenized_sentence]
    
    # Buat wadah kosong (array isi 0 semua) seukuran all_words
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0 # Tandai 1 jika kata ditemukan
            
    return bag

# --- TEMPAT NGETES KODE (Main) ---

# Tes Fungsi Tokenize
kalimat = "Gimana caranya join ke server?"
kata_kata = tokenize(kalimat)
print("1. Tokenize:", kata_kata)

# Tes Fungsi Stem
kata_dasar = [stem(w) for w in kata_kata]
print("2. Stemming:", kata_dasar)

# Tes Konsep Vektor (Bag of Words)
# Anggap ini kamus seluruh kata yang dimengerti bot (Vocabulary)
kamus_kata = ["bagaimana", "gimana", "cara", "join", "makan", "tidur", "server", "?"]

vektor = bag_of_words(kata_kata, kamus_kata)
print("3. Vektor Input:", vektor)
# Harusnya outputnya ada angka 1 di posisi kata yang cocok
import google.generativeai as genai
import sys
import os # Import untuk cek file

# --- 1. KONFIGURASI ---
API_KEY = "AIzaSyDh4fHIO2fbiYunYQV4kkO0dEx3xW1mpg4" 
genai.configure(api_key=API_KEY)

# --- 2. FUNGSI MEMBACA WIKI (LOADER) ---
def load_wiki_file(filename):
    try:
        # encoding='utf-8' penting biar emoji/simbol terbaca
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[ERROR] File '{filename}' tidak ditemukan!")
        return ""

# Baca file 'wiki.md' yang ada di folder yang sama
wiki_content = load_wiki_file("info_server.md")

# Gabungkan dengan instruksi kepribadian bot
system_instruction = f"""
ROLE:
Kamu adalah "NusantaraBot", asisten server Hytale.
Gunakan data berikut sebagai pengetahuan utamamu.

DATA WIKI SERVER:
{wiki_content}

ATURAN MENJAWAB:
- Jawab dengan santai dan ramah.
- Jika info tidak ada di wiki, bilang tidak tahu.
"""

# --- 3. INISIALISASI MODEL ---
# Masukkan system_instruction yang sudah digabung tadi
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=system_instruction
)

chat_session = model.start_chat(history=[])

# ... (Sisanya sama persis dengan kode sebelumnya)

# --- 4. LOOP CHAT UTAMA ---
def main():
    print("==========================================")
    print("🤖 NusantaraBot Siap Melayani 🤖")
    print("Ketik 'quit' atau 'exit' untuk keluar.")
    print("==========================================")

    while True:
        try:
            # Input User
            user_input = input("\nKamu: ")
            
            # Cek keluar
            if user_input.lower() in ["quit", "exit", "keluar"]:
                print("NusantaraBot: Bye bye! Jangan lupa tanya-tanya lagi ya!")
                break
            
            # Kosongkan input biar gak error
            if not user_input.strip():
                continue

            # Kirim ke Gemini (Bot sedang mengetik...)
            print("NusantaraBot (Thinking...): ", end="", flush=True)
            
            # Kirim pesan ke API
            response = chat_session.send_message(user_input)
            
            # Hapus tulisan 'Thinking...' dan tampilkan jawaban
            # (Teknik menghapus baris di terminal)
            sys.stdout.write("\r" + " " * 30 + "\r") 
            print(f"NusantaraBot: {response.text}")

        except Exception as e:
            print(f"\n[Error]: Koneksi terputus atau API bermasalah. ({e})")

if __name__ == "__main__":
    main()
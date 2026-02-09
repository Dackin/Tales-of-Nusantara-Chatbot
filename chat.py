import google.generativeai as genai
import discord
from dotenv import load_dotenv
import os

#load .env
load_dotenv()

# --- 1. KONFIGURASI ---
rpd = 0
max_rpd = 20

gemapi = 1
max_gemapi = 10

discord_token = os.getenv("discord_token")
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def cek_gemapi():
    global gemapi
    if gemapi == gemapi:
        gemapi = 1
    GEMINI_API_KEY = os.getenv("gemini_API" + str(gemapi))
    genai.configure(api_key=GEMINI_API_KEY)

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
chat_sessions = {}

# --- 4. EVENT: KETIKA BOT SIAP ---
@client.event
async def on_ready():
    print(f'✅ Bot berhasil login sebagai {client.user}')
    print('Siap melayani warga Nusantara!')

# --- 5. EVENT: KETIKA ADA PESAN MASUK ---
@client.event
async def on_message(message):

    global rpd, gemapi

    # PENTING: Jangan biarkan bot merespon dirinya sendiri (Looping)
    if message.author == client.user:
        return

    # Opsional: Bot hanya merespon jika di-mention atau di channel khusus
    if message.channel.name != "nusantara-bot":
        return

    # Ambil isi pesan user
    user_input = message.content
    channel_id = message.channel.id

    if rpd >= max_rpd:
        if gemapi == max_gemapi:
            gemapi = 1
        else:
            gemapi += 1
            rpd = 0
        
        cek_gemapi()
        print(f"[INFO] Berganti ke Gemini API Key #{gemapi}")

    # Cek apakah channel ini sudah punya sesi chat history?
    if channel_id not in chat_sessions:
        chat_sessions[channel_id] = model.start_chat(history=[])
    
    chat = chat_sessions[channel_id]

    try:
        # Tampilkan status "Bot is typing..." di Discord
        async with message.channel.typing():
            # Kirim ke Gemini
            response = chat.send_message(user_input)
            
            # Kirim balasan ke Discord
            await message.channel.send(response.text)

            rpd += 1
            
            print(f"[{message.author}] bertanya: {user_input}") # Log di terminal

    except Exception as e:
        print(f"Error: {e}")
        await message.channel.send("Maaf, sirkuit otakku lagi konslet. Coba lagi nanti ya.")

# --- 6. JALANKAN BOT ---
client.run(discord_token)
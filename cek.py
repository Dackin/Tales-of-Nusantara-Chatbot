import google.generativeai as genai

# Masukkan API Key kamu
API_KEY = "AIzaSyDh4fHIO2fbiYunYQV4kkO0dEx3xW1mpg4"
genai.configure(api_key=API_KEY)

print("Daftar Model yang Tersedia:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
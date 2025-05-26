import os
import requests
from dotenv import load_dotenv

# Charger la clé depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Vérifie que la clé est bien chargée
if not api_key:
    print("⚠️ Clé API Groq manquante.")
    exit()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3-70b-8192",
    "messages": [
        {"role": "user", "content": "Explique simplement ce qu'est une cataracte."}
    ]
}

response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload)

if response.status_code == 200:
    print("\n✅ Réponse du modèle :\n")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"❌ Erreur {response.status_code}: {response.text}")

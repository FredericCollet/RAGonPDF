import requests
import time

# URL de l'API locale Ollama
OLLAMA_URL = "http://192.168.1.123:11434/api/generate"

# Prompt de test — on va le répéter pour simuler un contexte lourd
chunk = "Voici une phrase test. " * 100  # chaque répétition ~6 tokens

# Prompt final (simule 2000+ tokens en prompt)
long_prompt = chunk * 10  # ajustable pour tester différentes tailles

payload = {
    "model": "gemma3",  # ou "gemma:7b", selon ce que tu veux tester
    "prompt": long_prompt,
    "stream": False,      # désactive le streaming pour simplifier
}

try:
    start = time.time()
    print("Envoi de la requête...")
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    duration = time.time() - start

    print(f"\nRéponse en {duration:.2f} secondes")

    if response.status_code == 200:
        print("\n✅ Succès :")
        print(response.json()["response"])
    else:
        print(f"\n❌ Erreur HTTP {response.status_code}:")
        print(response.text)

except requests.exceptions.ReadTimeout:
    print("\n⏱️ Timeout atteint ! La requête a dépassé la limite de 60s.")
except requests.exceptions.ConnectionError as e:
    print(f"\n🚫 Erreur de connexion : {e}")
except Exception as e:
    print(f"\n⚠️ Erreur inattendue : {e}")

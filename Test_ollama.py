import requests

# URL de base pour l'API Ollama
base_url = "http://localhost:11434"

# Endpoint pour lister les modèles disponibles
endpoint = "/api/tags"

# Effectuer la requête GET
response = requests.get(f"{base_url}{endpoint}")

# Afficher le résultat
if response.status_code == 200:
    print("Modèles disponibles:")
    print(response.json())
else:
    print(f"Erreur: {response.status_code}")
    print(response.text)
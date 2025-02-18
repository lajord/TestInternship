import ollama
import chromadb
import json
import hashlib
import os

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("bank_faqs")

def get_embedding(text):
    """Génère un embedding à partir d'un texte avec DeepSeek."""
    response = ollama.embeddings(model="deepseek-r1:7b", prompt=text) 
    return response["embedding"]

def index_with_deepseek_chroma(filepath):
    """Extrait et indexe des données depuis un fichier JSON ou TXT avec DeepSeek et stocke dans ChromaDB."""
    with open(filepath, "r", encoding="utf-8") as file:
        if filepath.endswith(".json"):
            documents = json.load(file)
        else:  # Fichier TXT
            documents = [{"content": section} for section in file.read().split("\n\n") if section.strip()]

    for doc in documents:
        content = doc.get("question", doc.get("content", ""))
        answer = doc.get("answer", "")
        doc_id = hashlib.md5(content.encode()).hexdigest()

        embedding = get_embedding(content)  # Générer un embedding avec DeepSeek

        # Stocker dans ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"content": content, "answer": answer}]
        )

def process_directory(directory):
    """Parcourt un répertoire et indexe les fichiers JSON et TXT avec DeepSeek + ChromaDB."""
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".json") or filename.endswith(".txt"):
            index_with_deepseek_chroma(filepath)

def generate_answer(query):
    """Recherche des documents avec DeepSeek dans ChromaDB et génère une réponse avec Ollama."""
    
    # 1. Obtenir l'embedding de la question avec DeepSeek
    query_embedding = get_embedding(query)

    # 2. Interroger ChromaDB pour récupérer les documents les plus pertinents
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  
    )

    # 3. Construire un contexte avec les résultats trouvés
    top_items = search_results["metadatas"][0] if search_results["metadatas"] else []
    context_parts = []
    
    for item in top_items:
        content = item.get("content", "")
        answer = item.get("answer", "")

        if answer:
            context_parts.append(f"Q: {content}\nA: {answer}")
        else:
            context_parts.append(f"Extrait: {content}")

    context_text = "\n\n".join(context_parts)

    # 4. Construire le prompt à envoyer à Ollama
    prompt = f"""
Voici des informations extraites de la FAQ et des documents :

{context_text}

En te basant sur ces informations, réponds à la question suivante de manière concise et claire :

Question utilisateur : {query}
Réponse :
    """

    # 5. Générer une réponse avec Ollama fine-tuné
    response = ollama.generate(
        model="model_finetune_3",
        prompt=prompt
    )

    return response.get("response", "Désolé, je n'ai pas trouvé d'information.")

if __name__ == "__main__":
    # Indexer les documents du dossier (exemple)
    # process_directory(r"H:\TestStage\DocumentIndexation")

    # Tester une requête utilisateur
    user_query = "Fait un top 10 jeux vidéox"
    answer = generate_answer(user_query)
    print("Réponse du modèle :", answer)

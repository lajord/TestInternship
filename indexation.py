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



process_directory("./DocumentIndexation")
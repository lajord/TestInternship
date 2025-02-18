import ollama
import chromadb


# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("bank_faqs")

def get_embedding(text):
    response = ollama.embeddings(model="deepseek-r1:7b", prompt=text)
    return response["embedding"]

def search_with_deepseek(query, query_embedding):
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  
    )

    top_items = search_results["metadatas"][0] if search_results["metadatas"] else []
    
    if not top_items:
        return ""  
    
    docs_text = "\n\n".join([f"Document {i+1}: {item.get('content', '')}" for i, item in enumerate(top_items)])
    
    deepseek_prompt = f"""
    You are an expert information retrieval assistant.
    Here are several documents retrieved from a database:

    {docs_text}

    Analyze them and select only the most relevant document to answer the user’s question.
    If none of the documents are relevant, return nothing.

    User question: {query}
    Response:
    """
    
    deepseek_response = ollama.chat(
        model="deepseek-r1:7b",
        messages=[{"role": "user", "content": deepseek_prompt}]
    )
    
    return deepseek_response.get("message", {}).get("content", "")

def generate_answer(query):
    query_embedding = get_embedding(query)
    selected_document = search_with_deepseek(query, query_embedding)
    
    if not selected_document.strip():
        return f"""
        No relevant document was found for the following question:
        User question: {query}
        Response: Sorry, I do not have enough information to answer this question.
        """
    
    prompt = f"""
    Here is a relevant document extracted from the database:

    {selected_document}

    Based on this information, provide a concise and clear answer to the following question:

    User question: {query}
    Response:
    """
    
    response = ollama.generate(
        model="model_finetune_3",
        prompt=prompt
    )
    
    return response.get("response", "Sorry, I could not find any relevant information.")

if __name__ == "__main__":
    # Initial introduction
    intro_prompt = "Introduce yourself briefly."
    intro_response = ollama.generate(
        model="model_finetune_3",
        prompt=intro_prompt
    )
    print("Bot:", intro_response.get("response", "Hello, I am your assistant!"), "\n")
    while True:
        user_query = input("Ask a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        print("Processing your request, please wait...\n")  
        answer = generate_answer(user_query)
        print("Response:", answer, "\n")
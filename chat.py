import ollama
import chromadb

# Connect to ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("bank_faqs")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    collection = None

def get_embedding(text):
    """Generate embedding using DeepSeek."""
    try:
        response = ollama.embeddings(model="deepseek-r1:7b", prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def search_with_deepseek(query, query_embedding):
    """Search for relevant documents using ChromaDB and DeepSeek."""
    if collection is None:
        return "Error: Database connection failed."
    
    try:
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5  
        )
        top_items = search_results.get("metadatas", [[]])[0]
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return ""
    
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
    
    try:
        deepseek_response = ollama.chat(
            model="deepseek-r1:7b",
            messages=[{"role": "user", "content": deepseek_prompt}]
        )
        return deepseek_response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"Error generating response from DeepSeek: {e}")
        return ""

def generate_answer(query):
    """Generate an answer using retrieved documents and fine-tuned model."""
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "Error: Failed to generate embedding."
    
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
    
    try:
        response = ollama.generate(
            model="model_finetune_3",
            prompt=prompt
        )
        return response.get("response", "Sorry, I could not find any relevant information.")
    except Exception as e:
        print(f"Error generating response from fine-tuned model: {e}")
        return "Sorry, I encountered an issue while processing your request."

if __name__ == "__main__":
    # Initial introduction
    try:
        intro_prompt = "Introduce yourself briefly."
        intro_response = ollama.generate(
            model="model_finetune_3",
            prompt=intro_prompt
        )
        print("Bot:", intro_response.get("response", "Hello, I am your assistant!"), "\n")
    except Exception as e:
        print(f"Error during bot introduction: {e}")
    
    while True:
        user_query = input("Ask a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        print("Processing your request, please wait...\n")  
        answer = generate_answer(user_query)
        print("Response:", answer, "\n")

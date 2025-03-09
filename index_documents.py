import os
from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel

# 📌 Cargar la API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# 📂 Carpeta con los PDFs
DOCUMENTS_DIR = "documentos"

# 🔄 Iniciar Embeddings y ChromaDB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 🚀 Inicializar FastAPI
app = FastAPI()

# 📄 Indexar los PDFs al iniciar
pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]

if not pdf_files:
    print("❌ No se encontraron PDFs para indexar.")
else:
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, pdf))
        docs = loader.load()
        vector_store.add_documents(docs)
        print(f"✅ {pdf} indexado en ChromaDB.")

    print("🚀 Indexación completada.")


# ✅ Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "Servidor corriendo 🚀"}

# 📌 Modelo para recibir consultas
class QueryModel(BaseModel):
    query: str

# 🔍 Endpoint para buscar en los documentos
@app.post("/query")
def search_documents(query_data: QueryModel):
    query = query_data.query
    if not query:
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")

    results = vector_store.similarity_search(query, k=5)  # Retorna los 5 más relevantes
    response = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]

    return {"query": query, "results": response}

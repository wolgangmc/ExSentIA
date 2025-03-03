import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader

# 📌 Cargar la API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# 📂 Carpeta con los PDFs
DOCUMENTS_DIR = "documentos"

# 🔄 Iniciar Embeddings y ChromaDB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 📄 Procesar cada PDF
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

import os
import chromadb
import openai
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# Inicializar el modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Conectar a la base de datos vectorial ChromaDB
chroma_client = chromadb.PersistentClient(path="vector_db")
collection = chroma_client.get_or_create_collection(name="exstream_docs")

def extraer_texto(archivo):
    """Lee el texto de archivos PDF, DOCX o TXT"""
    ext = archivo.split(".")[-1].lower()
    texto = ""
    
    if ext == "pdf":
        reader = PdfReader(archivo)
        for page in reader.pages:
            texto += page.extract_text() + "\n"
    
    elif ext == "docx":
        doc = Document(archivo)
        for para in doc.paragraphs:
            texto += para.text + "\n"
    
    elif ext == "txt":
        with open(archivo, "r", encoding="utf-8") as f:
            texto = f.read()
    
    return texto

# Procesar todos los documentos
documentos = os.listdir("documentos")
for archivo in documentos:
    ruta = os.path.join("documentos", archivo)
    print(f"Procesando: {archivo}")
    
    contenido = extraer_texto(ruta)
    fragmentos = [contenido[i:i+500] for i in range(0, len(contenido), 500)]  # Dividir en fragmentos
    
    for i, fragmento in enumerate(fragmentos):
        embedding = embedding_model.encode(fragmento).tolist()
        collection.add(ids=[f"{archivo}_{i}"], embeddings=[embedding], metadatas=[{"texto": fragmento}])

print("¡Documentos procesados y almacenados en la base de datos vectorial!")


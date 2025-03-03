import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings



# 🔥 Cargar variables de entorno
load_dotenv()

# 🔑 Obtener la API Key de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🚨 Validar si la API Key existe
if not openai_api_key:
    raise ValueError("❌ ERROR: No se encontró la API Key en las variables de entorno.")
else:
    print(f"✅ API Key detectada: {openai_api_key[:5]}******")  # Mostrar solo el inicio

# 📂 Carpeta donde están los PDFs
PDF_FOLDER = "documentos"

def load_pdfs():
    """ Carga todos los PDFs de la carpeta y extrae el texto. """
    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
                documents.extend(loader.load())
                print(f"📄 Cargado: {filename}")
            except Exception as e:
                print(f"⚠️ Error cargando {filename}: {e}")
    return documents

def create_vector_store():
    """ Convierte los textos en embeddings y los guarda en FAISS. """
    docs = load_pdfs()
    if not docs:
        raise ValueError("❌ ERROR: No se encontraron documentos en la carpeta.")

    texts = [doc.page_content for doc in docs]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    try:
        vectors = np.array(embeddings.embed_documents(texts)).astype("float32")
    except Exception as e:
        raise RuntimeError(f"❌ ERROR generando embeddings: {e}")

    # Crear FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Guardar la BD vectorial
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(index, f)
    with open("documents.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("✅ Base de datos vectorial creada y guardada correctamente.")

if __name__ == "__main__":
    create_vector_store()

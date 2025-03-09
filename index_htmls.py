import os
import chromadb
from bs4 import BeautifulSoup

# 📂 Ruta donde están los archivos HTML
html_dir = "/home/wolfgang/chatbot-telegram/chatbot-web/htmls"  # ❗ CAMBIA ESTO a la carpeta donde tienes los HTML

# 🔍 Inicializar ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")  # Guarda los datos en disco
collection = chroma_client.get_or_create_collection(name="html_docs")

# 🛠️ Función para extraer texto limpio de un HTML
def extraer_texto_html(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        return soup.get_text(separator=" ", strip=True)

# 🔄 Indexar todos los archivos HTML en ChromaDB
for filename in os.listdir(html_dir):
    if filename.endswith(".html"):
        file_path = os.path.join(html_dir, filename)
        texto = extraer_texto_html(file_path)

        # Agregar a la base vectorial
        collection.add(
            ids=[filename],  # Usa el nombre del archivo como ID
            documents=[texto]  # Guarda el texto extraído
        )

        print(f"✅ Indexado: {filename}")

print("🚀 Todos los archivos HTML han sido indexados en ChromaDB.")

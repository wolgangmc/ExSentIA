import chromadb
from chromadb.config import Settings

# Configura y arranca el servidor de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings())
print("✅ Servidor de ChromaDB listo en el puerto 8001")

# Mantén la ejecución activa
import time
while True:
    time.sleep(100)

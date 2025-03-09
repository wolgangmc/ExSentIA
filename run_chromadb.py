from chromadb.config import Settings
import chromadb

chroma_server = chromadb.HttpServer(Settings(
    persist_directory="chroma_db",  # Ruta donde tienes la base
    host="0.0.0.0",
    port=8000
))

chroma_server.run()

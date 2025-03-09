import chromadb

# 🔍 Conectar a la base de datos
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="html_docs")

# 🔎 Hacer una consulta
query = "Cómo generar un documento en Exstream"
resultados = collection.query(
    query_texts=[query],
    n_results=5  # Recuperar los 5 documentos más relevantes
)

# 📌 Mostrar resultados
for i, doc in enumerate(resultados["documents"][0]):
    print(f"🔹 Resultado {i+1}: {doc[:300]}...\n")

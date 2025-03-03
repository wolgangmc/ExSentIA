import os
import faiss
import pickle
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import openai

# 📌 Cargar la API Key de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ ERROR: No se encontró la API Key en Railway.")

openai.api_key = openai_api_key  # Asegurarse de que OpenAI la reconozca

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📂 Servir archivos estáticos (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# 📂 Cargar la base de datos vectorial
def load_vector_store():
    try:
        with open("faiss_index.pkl", "rb") as f:
            index = pickle.load(f)
        with open("documents.pkl", "rb") as f:
            texts = pickle.load(f)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS(index, embeddings, texts, index_to_docstore_id={i: i for i in range(len(texts))})
        retriever = vector_store.as_retriever()
        return retriever
    except FileNotFoundError:
        print("⚠️ No se encontró la base de datos vectorial. El bot responderá sin documentos.")
        return None

retriever = load_vector_store()

# 🔍 Generar respuestas basadas en la base vectorial
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")
chain = ConversationalRetrievalChain.from_llm(llm, retriever) if retriever else None

# 📩 Modelo de mensaje recibido
class Message(BaseModel):
    text: str

# 🎯 Endpoint de chat
@app.post("/chat")
async def chat(message: Message):
    """Recibe una pregunta y responde con la mejor opción disponible."""
    if chain:
        respuesta = chain.invoke({"question": message.text, "chat_history": []})
        return {"response": respuesta}
    else:
        return {"response": "⚠️ No hay base de datos cargada. Usa solo GPT-4."}

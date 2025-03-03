import os
import faiss
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Montar la carpeta 'static' para servir archivos HTML, CSS, JS
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# 📂 Cargar la base de datos vectorial
def load_vector_store():
    try:
        with open("faiss_index.pkl", "rb") as f:
            index = pickle.load(f)
        with open("documents.pkl", "rb") as f:
            texts = pickle.load(f)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS(index, embeddings, texts)
        retriever = vector_store.as_retriever()
        return retriever
    except FileNotFoundError:
        print("⚠️ No se encontró la base de datos vectorial. El bot responderá sin documentos.")
        return None

retriever = load_vector_store()

# 🔍 Generar respuestas basadas en la base vectorial
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")
chain = ConversationalRetrievalChain.from_llm(llm, retriever)

# 📩 Modelo de mensaje recibido
class Message(BaseModel):
    text: str

# 🎯 Endpoint de chat
@app.post("/chat")
async def chat(message: Message):
    """Recibe una pregunta y responde con la mejor opción disponible."""
    respuesta = chain.invoke(message.text)
    return {"response": respuesta}

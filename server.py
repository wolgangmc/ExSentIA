import os
import pickle
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma

import openai

# 📌 Cargar la API Key de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ ERROR: No se encontró la API Key en Railway.")

# Inicializar OpenAI
client = openai.OpenAI(api_key=openai_api_key)

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

# 📚 Servir archivos estáticos (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# 📂 Cargar la base de datos vectorial con ChromaDB
def load_vector_store():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        print("✅ Base de datos ChromaDB cargada correctamente.")
        return retriever
    except Exception as e:
        print(f"⚠️ No se pudo cargar ChromaDB. Usando solo GPT-4. Error: {e}")
        return None

# 🔍 Inicializar ChromaDB
retriever = load_vector_store()
if retriever is None:
    print("⚠️ No se pudo cargar ChromaDB. Usando solo GPT-4.")

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
    try:
        if chain:
            respuesta = chain.invoke({"question": message.text, "chat_history": []})
            return {"response": respuesta}
        else:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en OpenText Exstream."},
                    {"role": "user", "content": message.text}
                ]
            )
            return {"response": response.choices[0].message.content}
    except Exception as e:
        print(f"❌ Error en el chat: {e}")
        return {"response": "❌ Ocurrió un error. Inténtalo de nuevo más tarde."}

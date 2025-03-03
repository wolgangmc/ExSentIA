import os
import faiss
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
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
def generate_response_with_docs(question):
    """Usa la base vectorial si está disponible, de lo contrario responde con GPT."""
    if retriever:
        prompt_template = """Eres un asistente experto en OpenText Exstream. Responde la siguiente pregunta basándote en los documentos:
        {context}
        Pregunta: {question}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | OpenAIEmbeddings(openai_api_key=openai_api_key)
            | StrOutputParser()
        )
        return chain.invoke(question)
    else:
        return generate_response_gpt(question)

# 🔥 Generar respuestas solo con GPT si no hay base vectorial
def generate_response_gpt(question):
    """Consulta GPT-4 directamente cuando no hay documentos disponibles."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# 📩 Modelo de mensaje recibido
class Message(BaseModel):
    text: str

# 🎯 Endpoint de chat
@app.post("/chat")
async def chat(message: Message):
    """Recibe una pregunta y responde con la mejor opción disponible."""
    respuesta = generate_response_with_docs(message.text)
    return {"response": respuesta}

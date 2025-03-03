from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Cargar API Key de OpenAI desde .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# 🚀 NUEVA RUTA PARA EVITAR ERROR 404 EN RAILWAY
@app.get("/")
def read_root():
    return {"message": "¡El bot está corriendo correctamente!"}

# Permitir peticiones desde cualquier navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    """Recibe un mensaje y responde con GPT (compatible con OpenAI 1.0+)"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message.text}]
    )
    return {"response": response.choices[0].message.content}

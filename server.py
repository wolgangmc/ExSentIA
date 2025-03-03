from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Cargar API Key de OpenAI desde las variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verificar si la API Key está cargada correctamente
if not openai.api_key:
    raise ValueError("❌ ERROR: La API Key de OpenAI no está configurada. Verifica la variable de entorno en Railway.")

app = FastAPI()

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

@app.get("/")
async def root():
    return {"message": "¡Bienvenido al chatbot de ExSentIA!"}

@app.get("/env")
async def get_env():
    """Endpoint temporal para verificar si Railway está detectando la API Key"""
    return {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}

@app.post("/chat")
async def chat(message: Message):
    """Recibe un mensaje y responde con GPT (compatible con OpenAI 1.0+)"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message.text}]
        )
        return {"response": response.choices[0].message.content}
    
    except openai.OpenAIError as e:
        return {"error": f"❌ Error con OpenAI: {str(e)}"}
    
    except Exception as e:
        return {"error": f"❌ Error inesperado: {str(e)}"}

import os
from fastapi import FastAPI

app = FastAPI()

# 🔥 FORZAR CARGA Y DEBUG 🔥
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ ERROR: No se encontró la API Key en Railway.")
else:
    print(f"✅ API Key detectada en Railway: {openai_api_key[:5]}******")

@app.get("/env")
async def get_env():
    return {"OPENAI_API_KEY": openai_api_key}

import os
import json
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict

# ------------------- CONFIGURACIÓN NECESARIA -------------------
CARPETA_RAG = "./mi_rag"  # Carpeta donde se guardó Faiss y los chunks

# Configuración de la API de LM Studio (Puerto por defecto 1234)
LM_STUDIO_URL = "http://localhost:1234/v1" 

# Modelos (DEBEN estar corriendo como servidores en LM Studio)
NOMBRE_EMBEDDING_MODELO = "text-embedding-nomic-embed-text-v2-moe" 
NOMBRE_CHAT_MODELO = "openai/gpt-oss-20b"
# ---------------------------------------------------------------

# Conexión a la API de LM Studio
try:
    cliente_api = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
except Exception as e:
    print(f"❌ ERROR: No se pudo crear el cliente de OpenAI. ¿Está el servidor LM Studio corriendo? {e}")
    exit()

# ----------------------------------------
## 1. Carga del Índice y del Contexto
# ----------------------------------------
def cargar_base_de_datos(carpeta: str):
    """Carga el índice FAISS y los chunks de texto."""
    print("⏳ Cargando índice FAISS y chunks...")
    try:
        index = faiss.read_index(os.path.join(carpeta, "faiss.index"))
        with open(os.path.join(carpeta, "chunks.json"), "r", encoding="utf-8") as f:
            metadatos = json.load(f)
        print("✅ Base de datos cargada correctamente.")
        return index, metadatos
    except FileNotFoundError as e:
        print(f"❌ ERROR: Archivos FAISS o JSON no encontrados en {carpeta}. Ejecuta primero el script de indexación.")
        raise e

index, metadatos = cargar_base_de_datos(CARPETA_RAG)

# ----------------------------------------
## 2. Funciones de Búsqueda y Generación
# ----------------------------------------

def vectorizar_pregunta(pregunta: str) -> np.ndarray:
    """Convierte la pregunta en un vector usando el modelo Nomic (via LM Studio)."""
    try:
        # Pide a la API de LM Studio que vectorice la pregunta
        respuesta = cliente_api.embeddings.create(
            model=NOMBRE_EMBEDDING_MODELO,
            input=[pregunta]
        )
        # Retorna el primer y único embedding
        return np.array(respuesta.data[0].embedding, dtype='float32')
    except Exception as e:
        print(f"❌ ERROR al vectorizar la pregunta: {e}")
        return None

def buscar_contexto(vector_pregunta: np.ndarray, index: faiss.Index, metadatos: List[Dict], k: int = 4) -> str:
    """Busca los K chunks más relevantes en FAISS y forma el contexto."""
    
    # 1. Búsqueda FAISS: Distancias (D) e Índices (I)
    vector_pregunta = vector_pregunta.reshape(1, -1) # Formato requerido por Faiss
    D, I = index.search(vector_pregunta, k) 

    contextos_recuperados = []
    fuentes_usadas = set()

    # 2. Recuperar el texto original (chunks)
    for idx in I[0]:
        if idx >= 0 and idx < len(metadatos):
            chunk = metadatos[idx]
            contextos_recuperados.append(f"--- Fuente: {chunk['fuente']} ---\n{chunk['texto']}")
            fuentes_usadas.add(chunk['fuente'])
    
    # Unir todos los chunks relevantes en un solo string de contexto
    contexto_str = "\n\n".join(contextos_recuperados)
    
    print(f"\n🔍 Contexto recuperado de {len(contextos_recuperados)} fragmentos. Fuentes únicas: {list(fuentes_usadas)}")
    return contexto_str

def generar_respuesta(pregunta: str, contexto: str) -> str:
    """Envía la pregunta y el contexto al LLM (gpt-oss-20b) para generar la respuesta."""
    
    # 1. Crea el Prompt Aumentado (System Prompt + Contexto + Pregunta)
    prompt_sistema = (
        "Eres un asistente de respuesta de preguntas que utiliza la información proporcionada en el Contexto para responder "
        "concisa y con precisión. Si la respuesta no está en el Contexto, indica claramente que no tienes suficiente información. "
        "NO inventes información."
    )
    
    prompt_usuario = (
        f"Contexto:\n---\n{contexto}\n---\n\n"
        f"Pregunta: {pregunta}"
    )

    # 2. Llama a la API de Chat (LM Studio emula el endpoint de Chat de OpenAI)
    print("🤖 Generando respuesta con el LLM...")
    try:
        response = cliente_api.chat.completions.create(
            model=NOMBRE_CHAT_MODELO,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.1 # Baja temperatura para respuestas más fácticas
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ ERROR al comunicarse con el LLM '{NOMBRE_CHAT_MODELO}'. Asegúrate que esté corriendo en LM Studio: {e}"


# ----------------------------------------
## 3. Bucle Principal de Preguntas
# ----------------------------------------
def bucle_preguntas():
    print("\n" + "="*50)
    print("      Sistema RAG Local Listo. ¡Haz tu pregunta!")
    print(f"LLM: {NOMBRE_CHAT_MODELO} | Embeddings: {NOMBRE_EMBEDDING_MODELO}")
    print("Escribe 'salir' para terminar.")
    print("="*50 + "\n")

    while True:
        pregunta = input("❓ Tu pregunta: ").strip()
        
        if pregunta.lower() == 'salir':
            print("👋 ¡Adiós!")
            break
        if not pregunta:
            continue

        # 1. Vectorizar la pregunta
        vector_pre = vectorizar_pregunta(pregunta)
        if vector_pre is None:
            continue
        
        # 2. Buscar contexto en Faiss
        contexto = buscar_contexto(vector_pre, index, metadatos)
        
        # 3. Generar respuesta aumentada
        respuesta_final = generar_respuesta(pregunta, contexto)
        
        # 4. Mostrar resultado
        print("\n" + "-"*50)
        print("💡 Respuesta del LLM:")
        print(respuesta_final)
        print("-"*50 + "\n")

# Inicia el programa
bucle_preguntas()
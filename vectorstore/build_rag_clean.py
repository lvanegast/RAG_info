import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss

# Asegúrate de tener instalado: pip install openai numpy faiss-cpu tqdm PyPDF2

# ------------------- CONFIGURACIÓN NECESARIA -------------------
# Directorios
CARPETA_DOCUMENTOS = "./context_files"       # Aquí pones tus PDFs y TXT
CARPETA_RAG = "./mi_rag"                     # Carpeta donde se guardará Faiss y los chunks

# Modelo de LM Studio
# ¡IMPORTANTE! Este modelo debe estar corriendo como Servidor de Embeddings en LM Studio.
# Su nombre lo puedes ver en la pestaña 'Server' de LM Studio.
NOMBRE_MODELO_NOMIC = "text-embedding-nomic-embed-text-v2-moe" 

# Configuración de la API de LM Studio (Puerto por defecto 1234)
LM_STUDIO_URL = "http://localhost:1234/v1" 
# ---------------------------------------------------------------

# Creamos la carpeta de salida si no existe
os.makedirs(CARPETA_RAG, exist_ok=True)

# ----------------------------------------
## 1. Carga de Documentos
# ----------------------------------------
print("📚 Leyendo documentos de la carpeta:", CARPETA_DOCUMENTOS)
textos = []
fuentes = []

# Función para cargar TXT
def cargar_txt():
    for archivo in glob.glob(f"{CARPETA_DOCUMENTOS}/**/*.txt", recursive=True):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                textos.append(f.read())
                fuentes.append(os.path.relpath(archivo, CARPETA_DOCUMENTOS))
        except Exception as e:
            print(f"Error al leer TXT {archivo}: {e}")

# Función para cargar PDF
def cargar_pdf():
    try:
        from PyPDF2 import PdfReader
        for archivo in glob.glob(f"{CARPETA_DOCUMENTOS}/**/*.pdf", recursive=True):
            try:
                reader = PdfReader(archivo)
                texto = ""
                for page in reader.pages:
                    # Usamos .get('/Contents') para asegurar que extraiga contenido válido
                    texto += page.extract_text() + "\n"
                textos.append(texto)
                fuentes.append(os.path.relpath(archivo, CARPETA_DOCUMENTOS))
            except Exception as e:
                print(f"Error al procesar PDF {archivo}: {e}")
    except ImportError:
        print("ADVERTENCIA: No tienes PyPDF2. Solo se procesarán .txt.")
        print("Instálalo con: pip install PyPDF2")

cargar_txt()
cargar_pdf()

if not textos:
    print("❌ No se encontraron documentos para procesar. ¡Asegúrate de que 'context_files' no esté vacío!")
    exit()

# ----------------------------------------
## 2. División en Chunks (Fragmentación)
# ----------------------------------------
def dividir_texto(texto, tamano=1000, overlap=200):
    """Divide un texto largo en fragmentos con superposición."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano
        chunk = texto[inicio:fin]
        chunks.append(chunk)
        # Evita ir hacia atrás si el chunk es el último y más pequeño que el overlap
        inicio = max(fin - overlap, inicio + 1)
        if fin >= len(texto):
            break
    return chunks

todos_los_chunks = []
metadatos = []

print("✂️ Dividiendo documentos en chunks...")
for i, texto in enumerate(textos):
    # Aseguramos que el texto no esté vacío para evitar errores
    if not texto.strip():
        continue
        
    chunks = dividir_texto(texto, tamano=1000, overlap=200)
    for j, chunk in enumerate(chunks):
        todos_los_chunks.append(chunk)
        metadatos.append({
            "chunk_id": len(todos_los_chunks)-1,
            "fuente": fuentes[i],
            "texto": chunk.strip() # Guarda el texto real para la fase de consulta
        })

print(f"→ {len(todos_los_chunks)} chunks creados listos para vectorizar.")

# ----------------------------------------
## 3. Generar Embeddings (Usando LM Studio API)
# ----------------------------------------
from openai import OpenAI

try:
    # Conexión a la API de LM Studio
    cliente_api = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio") 
except Exception as e:
    print(f"❌ ERROR: No se pudo crear el cliente de OpenAI. ¿Está el servidor LM Studio corriendo? {e}")
    exit()

embeddings_list = []
BATCH_SIZE = 32 # Tamaño de lote recomendado para la API

print(f"🧠 Generando embeddings con LM Studio ({NOMBRE_MODELO_NOMIC})...")

# Generar embeddings por lotes
for i in tqdm(range(0, len(todos_los_chunks), BATCH_SIZE), desc="Vectorizando Chunks"):
    batch_chunks = todos_los_chunks[i:i + BATCH_SIZE]
    
    try:
        respuesta = cliente_api.embeddings.create(
            model=NOMBRE_MODELO_NOMIC,
            input=batch_chunks
        )
        
        # Extraer los vectores (embeddings) de la respuesta
        batch_embeddings = [data.embedding for data in respuesta.data]
        embeddings_list.extend(batch_embeddings)
        
    except Exception as e:
        print(f"\n❌ ERROR de API en lote {i}. Asegúrate que el modelo '{NOMBRE_MODELO_NOMIC}' esté corriendo en LM Studio.")
        print(f"Detalle del error: {e}")
        # En caso de error, terminamos y usamos solo los que pudimos generar (si hay alguno)
        break 

embeddings = np.array(embeddings_list, dtype='float32') 

if len(embeddings) == 0:
    print("❌ No se pudieron generar embeddings. Revisa la consola de LM Studio.")
    exit()

print(f"→ Embeddings generados: {embeddings.shape}")

# ----------------------------------------
## 4. Creación y Guardado del Índice FAISS
# ----------------------------------------
print("💾 Creando y guardando índice FAISS...")
dimension = embeddings.shape[1]
# IndexFlatL2 es un índice simple de distancia euclidiana (L2)
index = faiss.IndexFlatL2(dimension)
index.add(embeddings) # Agrega los embeddings al índice

# 5. Guardar todo en la carpeta de RAG
faiss.write_index(index, os.path.join(CARPETA_RAG, "faiss.index"))
with open(os.path.join(CARPETA_RAG, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(metadatos, f, ensure_ascii=False, indent=2)

print("\n🎉 ¡PROCESO DE INDEXACIÓN TERMINADO CON ÉXITO!")
print(f"Los archivos de RAG están en: {os.path.abspath(CARPETA_RAG)}")
print("   ├── faiss.index (Índice vectorial para búsqueda rápida)")
print("   └── chunks.json (Metadatos y texto original)")
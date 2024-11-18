import streamlit as st
import time
import os
from  background.bg_settings import render_title_and_background_buttons, render_sidebar
from src.loaders.load import load_documents_from_folder
from src.chunking.chunk import chunk_text
from src.embedding.embedding import CustomHuggingFaceEmbeddings
from src.vector_store_client.vstore import create_vector_store
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import OllamaLLM

##################################################################################
##Parametros:

chunk_size = 500
chunk_overlap = 100

# Carpeta de archivos para vstore
folder_path = "../practicos-rag/data"

# Carpeta de destino para los documentos subidos
UPLOAD_FOLDER = folder_path + '/new_data'

# Nombre de la colección
collection_name = "demo_collection2"
update = False
# Configuración de la página
st.set_page_config(page_title="Bienvenido, hazme una pregunta", layout="wide")

##################################################################################
##Funciones
#Carga de documentos:
docs_from_folder = load_documents_from_folder(folder_path)

#Chunking
splits = chunk_text(docs_from_folder, chunk_size, chunk_overlap)

# Crear una instancia de la clase personalizada
embedding_model = CustomHuggingFaceEmbeddings(mode="sentence")

# Obtener la dimensión del vector
dimension = embedding_model.get_dimension()
print(f"La dimensión del vector es: {dimension}")

# Conectar al cliente Qdrant
client = QdrantClient(host="localhost", port=6333)
print('Cliente conectado')

vector_store = create_vector_store(client, collection_name, embedding_model, splits, dimension, update)

#!ollama pull llama3.2
llm = OllamaLLM(model="llama3.2")


from langchain import hub
prompt = hub.pull("rlm/rag-prompt")
prompt.messages[0].prompt.template

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = vector_store.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
def rag_chain(retriever, prompt, llm):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser())
    return rag_chain

rag_chains = rag_chain(retriever, prompt, llm)
##################################################################################

# Renderizar la barra lateral
render_sidebar(UPLOAD_FOLDER)
render_title_and_background_buttons()

# Título de la página
#st.title("MODELO RAG")

# Cuadro de texto para la pregunta del usuario
query = st.text_input("Ingrese su pregunta:")

# Botón para ejecutar la inferencia
if st.button("Generar Respuesta"):
    if query:
        st.write(f"Consulta ingresada: {query}")

        # Cuadro para la respuesta generada
        response_placeholder = st.empty()  # Espacio para mostrar la respuesta en tiempo real
        # Respuesta final
        final_response = rag_chains.invoke(query)
        response_placeholder.write(f"**Respuesta generada:** {final_response}")
    else:
        st.warning("Por favor, ingrese una consulta antes de generar la respuesta.")


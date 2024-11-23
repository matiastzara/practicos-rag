import os
import streamlit as st
from dotenv import load_dotenv
from background.bg_settings import render_title_and_background_buttons, render_sidebar
from background.data_pipeline import initialize_data_pipeline, handle_user_query

# Cargar variables de entorno
load_dotenv()

##################################################################################
# Configuración de Streamlit
st.set_page_config(page_title="Chatbot RAG", layout="wide")

##################################################################################
# Parámetros iniciales
directory_path = "../practicos-rag/data/usa/"
UPLOAD_FOLDER = os.path.join(directory_path, 'new_data')

##################################################################################
# Inicializar elementos de la interfaz
render_sidebar(UPLOAD_FOLDER)
render_title_and_background_buttons()

# Manejo del historial de chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Inicializar RAG chain
rag_chain, retriever = initialize_data_pipeline(directory_path)

##################################################################################
# Interfaz de usuario: Chat
st.title("Chatbot RAG: Recuperación aumentada por generación")
user_input = st.chat_input("Escribe tu pregunta:")

if user_input:
    handle_user_query(user_input, retriever)
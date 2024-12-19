import streamlit as st

# Configurar la página como el primer comando
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

import logging
import os
from src.retrievers.retrievers import load_config
from background.streamlit_ui import (
    configure_ui, render_chat_interface, render_chat_history, render_model_selector,
    safe_initialize_rag, render_file_uploader)
from background.bgstyle import (
    render_title_and_background_buttons, apply_background_style
)

def main():
    logging.basicConfig(level=logging.INFO)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Configurar la interfaz
    configure_ui()
    
    # Aplicar el estilo de fondo al cargar
    apply_background_style()

    # Renderizar título y botones de fondo
    render_title_and_background_buttons()

    # Renderizar el cargador de archivos en la barra lateral
    render_file_uploader()

    # Selector de modelo (super o naive)
    render_model_selector("config.yaml")

    # Cargar configuración actualizada
    config = load_config('config.yaml')

    # Inicializar componentes RAG de forma segura
    rag_chain, retriever, chunks = safe_initialize_rag(config)

    # Guardar el modelo en session_state solo si es válido
    if rag_chain:
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever
        st.session_state.chunks = chunks

    # Renderizar historial de chat una sola vez
    st.container()  # Contenedor único para el historial
    render_chat_history()

    # Renderizar la interfaz principal del chatbot (sin historial duplicado)
    if rag_chain:
        render_chat_interface()  # Elimina el renderizado del historial aquí
    else:
        st.warning("El modelo no está disponible. La interfaz principal no puede ser utilizada.")

if __name__ == "__main__":
    main()

import streamlit as st
import logging
import os
from src.retrievers.retrievers import load_config
from src.background.streamlit_ui import (
    configure_ui, render_chat_interface, render_chat_history_with_scroll, render_model_selector,
    safe_initialize_rag, render_file_uploader, render_sidebar_image, render_evaluation_button)
from src.background.bgstyle import (render_title_and_background_buttons, apply_background_style)

def main():
    logging.basicConfig(level=logging.INFO)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Cargar configuración
    config = load_config("config.yaml")
    model_name = config.get("rag", "Modelo RAG")

    # Configurar la interfaz
    configure_ui()

    # Aplicar el estilo de fondo
    apply_background_style()

    # Renderizar botones de fondo
    render_title_and_background_buttons()

    # Renderizar barra lateral
    render_sidebar_image(image_path="src/background/miauc.png", caption="Modelo PinkBro")
    render_model_selector("config.yaml")
    render_file_uploader()
    render_evaluation_button(config)
    
    # Inicializar componentes RAG
    rag_chain, retriever, chunks = safe_initialize_rag(config)
    if rag_chain:
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever
        st.session_state.chunks = chunks

    # Inicializar historial de chat si no existe
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Renderizar historial de chat (una sola vez)
    render_chat_history_with_scroll()

    # Renderizar interfaz de entrada de chat
    if rag_chain:
        render_chat_interface()
    else:
        st.warning("El modelo no está disponible.")

if __name__ == "__main__":
    main()

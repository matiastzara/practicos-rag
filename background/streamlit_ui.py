import streamlit as st
import yaml
import os
from src.retrievers.retrievers import initialize_rag
from src.chunking.chunking import show_chunks_streamlit


def configure_ui():
    """Configura la interfaz inicial de Streamlit, incluyendo el historial de chat."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        
def process_user_query():
    """
    Procesa la consulta del usuario y actualiza el historial.
    """
    query = st.session_state.get("query", "").strip()
    if not query:
        st.warning("Por favor, ingresa una pregunta válida.")
        return

    rag_chain = st.session_state.get("rag_chain")
    if not rag_chain:
        st.error("El modelo RAG no está inicializado.")
        return

    # Generar respuesta usando el modelo RAG
    response = rag_chain.invoke(query)

    # Actualizar el historial
    st.session_state["chat_history"].append({"question": query, "response": response})

    # Limpiar el campo de entrada
    st.session_state["query"] = ""

    # Renderizar historial actualizado
    render_chat_history_with_scroll()

def render_chat_interface():
    """Renderiza solo la barra de entrada de texto para interactuar con el chatbot."""
    st.text_input(
        "Haz tu pregunta:",
        placeholder="Escribe tu consulta aquí...",
        on_change=process_user_query,
        key="query",
    )

def render_model_selector(config_path="config.yaml"):
    """Renderiza un selector de modelo (super o naive) y actualiza el archivo de configuración."""
    st.sidebar.markdown("### Selección de Modelo")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    selected_model = st.sidebar.radio(
        "Selecciona el modelo RAG a utilizar:",
        options=["super", "naive"],
        index=0 if config.get("rag") == "super" else 1,
    )

    if selected_model != config.get("rag"):
        config["rag"] = selected_model
        with open(config_path, "w") as file:
            yaml.safe_dump(config, file)
        st.success(f"Configuración actualizada: ahora se usa el modelo '{selected_model}'.")


def render_file_uploader(upload_folder="../practicos-rag/data"):
    """Renderiza la bandeja de carga de archivos en la barra lateral y guarda los archivos."""
    st.sidebar.image(
        "background/miauc.png",
        caption="Modelo RAG",
        use_container_width=True,
    )
    st.sidebar.header("Subir documentos")
    uploaded_files = st.sidebar.file_uploader(
        "Selecciona documentos para cargar:",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    if st.sidebar.button("Subir documentos"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Se subieron {len(uploaded_files)} archivo(s) correctamente a '{upload_folder}'.")
        else:
            st.sidebar.warning("No se seleccionaron archivos para cargar.")


def safe_initialize_rag(config):
    """Inicializa los componentes RAG de manera segura y los almacena en session_state."""
    if "rag_chain" not in st.session_state:
        try:
            rag_chain, retriever, chunks = initialize_rag(config)
            st.session_state["rag_chain"] = rag_chain
            st.session_state["retriever"] = retriever
            st.session_state["chunks"] = chunks
            st.success("Modelo RAG cargado exitosamente.")
        except Exception as e:
            st.error(f"Error al cargar el modelo RAG: {e}")
            st.session_state["rag_chain"] = None
            st.session_state["retriever"] = None
            st.session_state["chunks"] = []
    return (
        st.session_state.get("rag_chain"),
        st.session_state.get("retriever"),
        st.session_state.get("chunks"),
    )


# def render_chat_history():
#     """Renderiza el historial de chats en la interfaz con íconos para usuario y chatbot."""
#     st.markdown("### Historial de Chat")
#     for chat in st.session_state["chat_history"]:
#         st.markdown(f"🙂 **Usuario**: {chat['question']}")
#         st.markdown(f"🤖 **Chatbot**: {chat['response']}")

def render_chat_interface():
    """
    Renderiza la interfaz principal del chatbot, excluyendo el historial.
    """
    st.text_input(
        "Haz tu pregunta:",
        placeholder="Escribe tu consulta aquí...",
        on_change=process_user_query,
        key="query",
    )

    
def render_chat_history_with_scroll():
    """
    Renderiza el historial de chat en un bloque fijo con scroll.
    """
    st.markdown(
        """
        <style>
        .chat-container {
            height: 300px; /* Altura fija para el historial */
            overflow-y: auto; /* Barra de desplazamiento vertical */
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 8px;
            background-color: #2f2f2f; /* Fondo oscuro */
        }
        .chat-message {
            margin-bottom: 10px;
            padding: 5px;
            border-radius: 5px;
        }
        .chat-message span {
            font-weight: bold;
        }
        .user-message {
            color: #ffc107; /* Color amarillo para el usuario */
        }
        .bot-message {
            color: #17a2b8; /* Color azul para el chatbot */
        }
        </style>
        <div class="chat-container">
        """,
        unsafe_allow_html=True,
    )

    for chat in st.session_state.get("chat_history", []):
        user_message = f"<div class='chat-message user-message'><span>🙂 Usuario:</span> {chat['question']}</div>"
        bot_message = f"<div class='chat-message bot-message'><span>🤖 Chatbot:</span> {chat['response']}</div>"
        st.markdown(user_message + bot_message, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



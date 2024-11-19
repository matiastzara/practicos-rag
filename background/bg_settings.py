import streamlit as st
from background.file_upload import render_file_uploader

def initialize_background_state():
    """Inicializa el estado del fondo en st.session_state."""
    if "background" not in st.session_state:
        st.session_state["background"] = "white"  # Fondo blanco por defecto

def apply_background_style():
    """Aplica el estilo de fondo basado en el estado actual."""
    if st.session_state["background"] == "gray":
        set_background_gray()
    else:
        set_background_white()

def change_background(color):
    """Cambia el estado del fondo en session_state."""
    st.session_state["background"] = color

def set_background_gray():
    """Estilo para fondo gris, letras blancas y cuadros negros."""
    css = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #2f2f2f; /* Fondo gris oscuro */
        color: white; /* Letras blancas */
    }
    [data-testid="stSidebar"], input, textarea {
        background-color: black; /* Cuadros negros */
        color: white; /* Texto en cuadros */
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def set_background_white():
    """Estilo para fondo blanco, letras negras y cuadros gris claro."""
    css = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: white; /* Fondo blanco */
        color: black; /* Letras negras */
    }
    [data-testid="stSidebar"], input, textarea {
        background-color: #f0f0f0; /* Cuadros gris claro */
        color: black; /* Texto en cuadros */
    }
    [data-testid="stSidebar"] * {
        color: black;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_title_and_background_buttons():
    """Renderiza el título de la página y los botones para cambiar el fondo."""
    st.title("MODELO RAG")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fondo Gris"):
            change_background("gray")
            apply_background_style()  
    with col2:
        if st.button("Fondo Blanco"):
            change_background("white")
            apply_background_style()  

def render_sidebar(upload_folder):
    """Renderiza la barra lateral con logo, carga de archivos y chats creados."""
    # Logo desde una imagen local
    st.sidebar.image(
        "background/miauc.png",  # Ruta local al logo
        caption="Modelo RAG",
        use_container_width=True,
    )

    # Sección de carga de archivos
    st.sidebar.header("Cargar Nuevos Archivos")
    render_file_uploader(upload_folder)  # Reutiliza la función para cargar archivos

    # Chats creados
    st.sidebar.header("Chats Creados")
    if "chats" not in st.session_state:
        st.session_state["chats"] = []  # Inicializar lista de chats

    for idx, chat in enumerate(st.session_state["chats"]):
        if st.sidebar.button(chat):
            st.session_state["current_chat"] = chat  # Establecer chat actual

    # Botón para agregar nuevo chat
    if st.sidebar.button("Nuevo Chat"):
        new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
        st.session_state["chats"].append(new_chat_name)
        st.session_state["current_chat"] = new_chat_name
        
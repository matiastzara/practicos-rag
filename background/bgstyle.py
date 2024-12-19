import streamlit as st

def apply_background_style():
    """Aplica el estilo de fondo basado en el estado actual."""
    if "background" not in st.session_state:
        st.session_state["background"] = "gray"
    if st.session_state["background"] == "gray":
        set_background_gray()
    else:
        set_background_white()

def change_background(color):
    """Cambia el estado del fondo en session_state."""
    st.session_state["background"] = color

def set_background_gray():
    """Estilo para fondo gris cálido, letras contrastantes y cuadros estilizados."""
    css = """
    <style>
    /* Fondo principal */
    [data-testid="stAppViewContainer"] {
        background-color: #2A2A2A; /* Fondo gris cálido */
        color: #E0E0E0; /* Texto gris claro */
        font-family: 'Roboto', sans-serif; /* Fuente moderna */
        font-size: 14px; /* Tamaño de letra */
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #1F1F1F; /* Fondo gris más oscuro */
        color: #E0E0E0; /* Texto claro */
        border-right: 1px solid #4A4A4A; /* Borde para separar la barra */
    }

    /* Inputs y textarea */
    input, textarea {
        background-color: #333333; /* Fondo de cuadros */
        color: white; /* Texto claro */
        border: 1px solid #555555; /* Borde gris */
        border-radius: 8px; /* Bordes redondeados */
        padding: 8px; /* Espaciado interno */
    }

    /* Texto dentro de la barra lateral */
    [data-testid="stSidebar"] * {
        color: #E0E0E0;
        font-size: 14px;
    }

    /* Botones */
    button {
        background-color: #444444; /* Fondo del botón */
        color: white; /* Texto */
        border: 1px solid #555555; /* Borde */
        border-radius: 8px; /* Bordes redondeados */
        padding: 10px 15px;
        font-weight: bold;
    }

    button:hover {
        background-color: #555555; /* Cambio de color al pasar el mouse */
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def set_background_white():
    """Estilo para fondo gris claro cálido, letras negras suaves y cuadros estilizados."""
    css = """
    <style>
    /* Fondo principal */
    [data-testid="stAppViewContainer"] {
        background-color: #F7F7F7; /* Fondo gris claro cálido */
        color: #333333; /* Texto negro suave */
        font-family: 'Roboto', sans-serif; /* Fuente moderna */
        font-size: 14px; /* Tamaño de letra */
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #ECECEC; /* Fondo gris muy claro */
        color: #333333; /* Texto oscuro */
        border-right: 1px solid #D0D0D0; /* Borde claro */
    }

    /* Inputs y textarea */
    input, textarea {
        background-color: #FFFFFF; /* Fondo blanco */
        color: #333333; /* Texto negro suave */
        border: 1px solid #CCCCCC; /* Borde gris claro */
        border-radius: 8px; /* Bordes redondeados */
        padding: 8px; /* Espaciado interno */
    }

    /* Texto dentro de la barra lateral */
    [data-testid="stSidebar"] * {
        color: #333333;
        font-size: 14px;
    }

    /* Botones */
    button {
        background-color: #D0D0D0; /* Fondo del botón */
        color: black; /* Texto */
        border: 1px solid #B0B0B0; /* Borde */
        border-radius: 8px; /* Bordes redondeados */
        padding: 10px 15px;
        font-weight: bold;
    }

    button:hover {
        background-color: #C0C0C0; /* Cambio de color al pasar el mouse */
        color: black;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



# def render_title_and_background_buttons():
#     """Renderiza el título de la página y los botones para cambiar el fondo."""
#     col1, col2, col3 = st.columns([4, 1, 1])  # Distribución de columnas: título y botones
#     with col1:
#         st.title("MODELO RAG")
#     with col2:
#         if st.button("Fondo Gris", key="gray_button"):
#             change_background("gray")
#             apply_background_style()
#     with col3:
#         if st.button("Fondo Blanco", key="white_button"):
#             change_background("white")
#             apply_background_style()
            

def render_title_and_background_buttons():
    """Renderiza el título centrado y botones para cambiar el fondo."""
    css = """
    <style>
    .header-container {
        position: fixed;
        top: 60px; /* Ajustar para que no quede solapado */
        left: 0;
        width: 100%;
        background-color: #2A2A2A;
        z-index: 1000;
        padding: 10px 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
    }
    .header-container h1 {
        text-align: center;
        color: white;
        font-size: 24px;
    }
    .header-buttons {
        position: absolute;
        top: 50%;
        right: 10px;
        transform: translateY(-50%);
        display: flex;
        gap: 10px;
    }
    .header-buttons button {
        background-color: #444;
        color: white;
        border: 1px solid #555;
        border-radius: 8px;
        padding: 5px 10px;
        font-weight: bold;
        cursor: pointer;
    }
    .header-buttons button:hover {
        background-color: #555;
    }
    .main-content {
        margin-top: 120px;
    }
    </style>
    <script>
    function setBackgroundGray() {
        document.body.style.backgroundColor = "#2f2f2f";
        document.body.style.color = "white";
    }
    function setBackgroundWhite() {
        document.body.style.backgroundColor = "#f9f9f9";
        document.body.style.color = "black";
    }
    </script>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Renderizado del header con Streamlit
    col1, col2, col3 = st.columns([4, 1, 1])  # Distribución de columnas
    with col1:
        st.markdown('<h1 style="text-align: center; color: white;">MODELO RAG</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("Fondo Gris", key="gray_button"):
            change_background("gray")  # Cambiar el fondo a gris
            apply_background_style()
    with col3:
        if st.button("Fondo Blanco", key="white_button"):
            change_background("white")  # Cambiar el fondo a blanco
            apply_background_style()
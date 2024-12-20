import streamlit as st
import yaml
import os
from src.retrievers.retrievers import initialize_rag
from src.chunking.chunking import show_chunks_streamlit
from src.evaluation.evaluation import evaluate_and_save_results


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
    config = st.session_state.get("config", {})
    if not rag_chain:
        st.error("El modelo RAG no está inicializado.")
        return

    # Generar respuesta usando el modelo RAG
    response = rag_chain.invoke(query)
    model_used = config.get("rag", "Desconocido")  # Determina el modelo (naive o super)

    # Actualizar el historial
    st.session_state["chat_history"].append({
        "question": query,
        "response": response,
        "model": model_used  # Agrega el modelo usado
    })

    # Limpiar el campo de entrada
    st.session_state["query"] = ""

    # Renderizar historial actualizado
    render_chat_history_with_scroll()

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

def render_chat_history_with_scroll():
    """
    Renderiza el historial de chat en un formato conversacional con íconos.
    Muestra un mensaje inicial solo si el historial está vacío.
    """
    st.markdown("### Historial de Chat")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            if entry["sender"] == "user":
                st.markdown(f"👤 **Tú**: {entry['message']}")
            elif entry["sender"] == "bot":
                st.markdown(f"🤖 **Bot**: {entry['message']}")
    else:
        st.markdown("No hay historial de chat disponible.")  # Solo aparece si está vacío

            
def render_chat_interface():
    """
    Maneja la interacción del usuario, muestra la pregunta inmediatamente,
    y luego genera la respuesta del modelo.
    """
    # Entrada del usuario
    user_input = st.chat_input("Escribe tu mensaje...")  # Entrada interactiva del usuario
    if user_input:
        # Inicializar el historial si no existe
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Agregar la pregunta al historial
        st.session_state.chat_history.append({"sender": "user", "message": user_input})

        # Renderizar el historial inmediatamente con la pregunta añadida
        render_chat_history_with_scroll()

        # Crear un marcador temporal para la respuesta del bot
        bot_response_placeholder = st.empty()
        bot_response_placeholder.markdown("🤖 **Bot**: Procesando...")

        # Generar la respuesta del modelo
        try:
            response = st.session_state.rag_chain.invoke(user_input)
        except Exception as e:
            response = f"Error al generar respuesta: {e}"

        # Agregar la respuesta al historial
        st.session_state.chat_history.append({"sender": "bot", "message": response})

        # Actualizar el marcador con la respuesta final
        bot_response_placeholder.markdown(f"🤖 **Bot**: {response}")



def render_evaluation_button(config):
    """
    Renderiza el botón de evaluación y ejecuta la evaluación si es presionado.
    """
    st.sidebar.markdown("### Evaluación del modelo")
    if st.sidebar.button("Evaluar modelo"):
        if "rag_chain" in st.session_state and "retriever" in st.session_state:
            st.markdown("### Evaluando el modelo...")
            try:
                evaluation_results = evaluate_and_save_results(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    config,
                )
                st.markdown("### Resultados de la evaluación")
                st.dataframe(evaluation_results)
            except Exception as e:
                st.error(f"Error durante la evaluación: {e}")
        else:
            st.warning("El modelo no está disponible para evaluación.")


def render_sidebar_image(image_path="background/miauc.png", caption="Modelo RAG"):
    """
    Renderiza una imagen en la barra lateral con un título opcional.
    
    Args:
        image_path (str): Ruta al archivo de imagen.
        caption (str): Texto que aparecerá como pie de foto.
    """
    st.sidebar.image(
        image_path,
        caption=caption,
        use_container_width=True,
    )
    
def render_file_uploader(upload_folder="../practicos-rag/data"):
    """
    Renderiza la bandeja de carga de archivos en la barra lateral y gestiona la subida de archivos.
    
    Args:
        upload_folder (str): Ruta donde se guardarán los archivos subidos.
    """
    st.sidebar.header("Subir documentos")
    uploaded_files = st.sidebar.file_uploader(
        "Selecciona documentos para cargar:",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    # Crear la carpeta si no existe
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Botón para subir archivos
    if st.sidebar.button("Subir documentos"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Se subieron {len(uploaded_files)} archivo(s) correctamente a '{upload_folder}'.")
        else:
            st.sidebar.warning("No se seleccionaron archivos para cargar.")



# def render_file_uploader(upload_folder="../practicos-rag/data"):
#     """Renderiza la bandeja de carga de archivos en la barra lateral y guarda los archivos."""
#     st.sidebar.image(
#         "background/miauc.png",
#         caption="Modelo RAG",
#         use_container_width=True,
#     )
#     st.sidebar.header("Subir documentos")
#     uploaded_files = st.sidebar.file_uploader(
#         "Selecciona documentos para cargar:",
#         type=["pdf", "txt", "docx"],
#         accept_multiple_files=True,
#     )

#     if not os.path.exists(upload_folder):
#         os.makedirs(upload_folder)

#     if st.sidebar.button("Subir documentos"):
#         if uploaded_files:
#             for uploaded_file in uploaded_files:
#                 file_path = os.path.join(upload_folder, uploaded_file.name)
#                 with open(file_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
#             st.sidebar.success(f"Se subieron {len(uploaded_files)} archivo(s) correctamente a '{upload_folder}'.")
#         else:
#             st.sidebar.warning("No se seleccionaron archivos para cargar.")
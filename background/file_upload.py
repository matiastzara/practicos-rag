import os
import streamlit as st

def render_file_uploader(upload_folder):
    """Renderiza la bandeja de carga de archivos y guarda los archivos en la carpeta."""
    st.header("Subir documentos")
    uploaded_files = st.file_uploader(
        "Selecciona documentos para cargar",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    # Crear la carpeta si no existe
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Botón para procesar los archivos
    if st.button("Subir documentos"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Guardar el archivo en la carpeta de destino
                file_path = os.path.join(upload_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Se subieron {len(uploaded_files)} archivo(s) correctamente a '{upload_folder}'.")
        else:
            st.warning("No se seleccionaron archivos para cargar.")

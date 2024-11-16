from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from PyPDF2 import PdfReader
import os

# Función para enriquecer metadata de PDFs
def enrich_metadata(file_path):
    """
    Extrae metadata adicional de un archivo PDF usando PyPDF2.
    """
    reader = PdfReader(file_path)
    doc_metadata = reader.metadata
    enriched_metadata = {
        "author": doc_metadata.get("/Author", "Unknown"),
        "title": doc_metadata.get("/Title", "Unknown"),
        "creation_date": doc_metadata.get("/CreationDate", "Unknown"),
        "modification_date": doc_metadata.get("/ModDate", "Unknown"),
    }
    return enriched_metadata

# Función para cargar documentos desde una carpeta
def load_documents_from_folder(folder_path):
    """
    Carga documentos desde una carpeta usando loaders específicos según el tipo de archivo.
    """
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredFileLoader,
    }
    all_docs = []

    # Recorrer todos los archivos en la carpeta
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        _, file_extension = os.path.splitext(file_name)

        # Seleccionar el loader según la extensión del archivo
        loader_class = loaders.get(file_extension.lower())
        if loader_class:
            loader = loader_class(file_path)
            docs = loader.load()

            # Si es un PDF, enriquecer metadata
            if file_extension.lower() == ".pdf":
                enriched_metadata = enrich_metadata(file_path)
                for doc in docs:
                    doc.metadata.update(enriched_metadata)

            # Agregar los documentos a la lista
            all_docs.extend(docs)

    return all_docs
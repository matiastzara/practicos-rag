#Funcion para crear chunk mediante recursividad de caracteres y aplica limpieza para mejorar los embeddings:
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def clean_text(text):
    """Limpia el texto eliminando espacios redundantes y caracteres especiales al principio o al final."""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(docs, chunk_size=1000, chunk_overlap=200):
    """
    Divide los documentos en fragmentos con superposición opcional y limpia el contenido directamente.

    Args:
        docs (list): Lista de documentos en formato {'page_content': texto, ...}.
        chunk_size (int): Tamaño máximo de cada fragmento.
        chunk_overlap (int): Cantidad de superposición entre fragmentos.

    Returns:
        list: Lista de fragmentos de texto con contenido limpio.
    """
    # Inicializar el divisor de texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Dividir documentos en fragmentos
    splits = text_splitter.split_documents(docs)
    
    # Limpiar directamente el contenido de cada fragmento
    for split in splits:
        split.page_content = clean_text(split.page_content)
    
    return splits
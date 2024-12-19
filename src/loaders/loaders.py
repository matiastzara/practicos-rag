from dotenv import load_dotenv  

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  


def load_pdf(file_path):
    """
    Carga un archivo PDF y devuelve las primeras páginas como documentos.

    Args:
        file_path (str): Ruta al archivo PDF.

    Returns:
        list: Lista de documentos extraídos del PDF.
    """
    load_dotenv()

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    return docs  

def split_pdf_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Divide un documento PDF en fragmentos de texto.

    Args:
        docs (list): Lista de documentos cargados desde un PDF.
        chunk_size (int): Tamaño de cada fragmento de texto en caracteres. Default es 1000.
        chunk_overlap (int): Cantidad de solapamiento entre fragmentos. Default es 200.

    Returns:
        list: Lista de fragmentos de texto extraídos del PDF.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    return splits  

import os
from PyPDF2 import PdfReader

def load_pdf_all_documents(directory_path: str):
    """
    Carga documentos PDF desde una carpeta y devuelve una lista de páginas como texto.
    Args:
        directory_path (str): Ruta de la carpeta que contiene los archivos PDF.
    Returns:
        List[str]: Lista de textos extraídos de cada página de todos los PDFs en la carpeta.
    """
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            print(f"Cargando archivo: {file_path}")
            reader = PdfReader(file_path)
            for page in reader.pages:
                all_texts.append(page.extract_text())
    return all_texts

def clean_text_and_exclude_sections(text: str):
    """
    Limpia el texto eliminando espacios redundantes y caracteres especiales al principio o al final.
    Args:
        text (str): Texto extraído del PDF.
    Returns:
        str: Texto limpio.
    """
    import re
    # Reemplazar múltiples espacios consecutivos con un único espacio
    text = re.sub(r'\s+', ' ', text).strip()
    # Eliminar espacios al inicio y final del texto
    return text

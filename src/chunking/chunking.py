import pandas as pd
import os  
import re  
import logging  
from typing import List, Dict, Tuple, Optional  
from PyPDF2 import PdfReader  


def load_pdf_all_documents(directory_path: str) -> List[str]:
    """
    Carga documentos PDF desde una carpeta y devuelve una lista de páginas como texto.

    Args:
        directory_path (str): Ruta de la carpeta que contiene los archivos PDF.

    Returns:
        List[str]: Lista de cadenas de texto, donde cada cadena corresponde al texto extraído de una página PDF.
    """

    all_texts = list()
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            try:
                logging.info(f"Cargando archivo: {file_path}")
                reader = PdfReader(file_path)
                for page in reader.pages:
                    all_texts.append(page.extract_text())
            except Exception as e:
                logging.error(f"Error leyendo el archivo PDF: {file_path}. Detalle: {e}")
    return all_texts

def clean_text_and_exclude_sections(text: str) -> str:
    """
    Limpia el texto eliminando espacios redundantes y caracteres especiales.

    Args:
        text (str): Texto a limpiar.

    Returns:
        str: Texto limpio con espacios redundantes eliminados.
    """
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

def split_text_into_sentences(text: str) -> List[Dict[str, str]]:
    """
    Divide un texto en oraciones basado en '.', '?', y '!' y devuelve una lista de diccionarios.
    Args:
        text (str): El texto a dividir.
    Returns:
        List[Dict[str, str]]: Lista de diccionarios con 'sentence' y 'index'.
    """
    single_sentences_list = re.split(r'(?<=[.?!])\s+', text.strip())
    sentences = [{'sentence': sentence, 'index': i} for i, sentence in enumerate(single_sentences_list)]
    return sentences

def combine_sentences(sentences: List[Dict[str, str]], buffer_size: int = 1) -> List[Dict[str, str]]:
    """
    Combina oraciones de acuerdo al tamaño del buffer definido.
    Args:
        sentences (List[Dict[str, str]]): Lista de oraciones con índices.
        buffer_size (int): Número de oraciones antes y después a combinar.
    Returns:
        List[Dict[str, str]]: Lista con oraciones combinadas.
    """
    for i in range(len(sentences)):
        combined_sentence = ''

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']


        sentences[i]['combined_sentence'] = combined_sentence.strip()

    return sentences

def extract_metadata(text_chunk: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extrae títulos, subtítulos y sub-subtítulos de un fragmento de texto.

    Args:
        text_chunk (str): Fragmento de texto del cual extraer la metadata.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: Título, subtítulo y sub-subtítulo encontrados (o None si no se encuentran).
    """
    title_pattern = re.compile(r"PART \d+[-—]\s*[A-Za-z0-9 ,.\-]+")
    subtitle_pattern = re.compile(r"Subpart [A-Z]—[A-Za-z0-9 ,\\-]+")
    sub_subtitle_pattern = re.compile(r"§\s*\d+\.\d+\s+[A-Za-z0-9 ,.\-]+")

    title = title_pattern.search(text_chunk)
    subtitle = subtitle_pattern.search(text_chunk)
    sub_subtitle = sub_subtitle_pattern.search(text_chunk)

    return (
        title.group(0).strip() if title else None,
        subtitle.group(0).strip() if subtitle else None,
        sub_subtitle.group(0).strip() if sub_subtitle else None,
    )

def assign_metadata_to_chunks_with_context(chunks: List[str], max_previous_chunks: int = 100) -> List[Dict[str, str]]:
    """
    Asigna títulos, subtítulos y sub-subtítulos como metadata a cada chunk.

    Args:
        chunks (List[str]): Lista de fragmentos de texto.
        max_previous_chunks (int): Número máximo de fragmentos previos a considerar para acumular metadata.

    Returns:
        List[Dict[str, str]]: Lista de fragmentos con metadata asignada.
    """
    annotated_chunks = list()
    for i in range(len(chunks)):
        metadata_accumulated = {"title": None, "subtitle": None, "sub_subtitle": None}
        for j in range(max(0, i - max_previous_chunks), i):
            metadata_title, metadata_subtitle, metadata_subsubtitle = extract_metadata(chunks[j])
            if metadata_title:
                metadata_accumulated["title"] = metadata_title
            if metadata_subtitle:
                metadata_accumulated["subtitle"] = metadata_subtitle
            if metadata_subsubtitle:
                metadata_accumulated["sub_subtitle"] = metadata_subsubtitle
        annotated_chunks.append({"chunk_text": chunks[i], "metadata": metadata_accumulated.copy()})
    return annotated_chunks

def show_chunks_streamlit(chunks, config):
    """
    Esta función toma una lista de chunks y un diccionario de configuración.
    Guarda un número específico de los últimos chunks, nombrado según un parámetro en el config.

    Args:
    chunks (list): Lista de datos para convertir en DataFrame y guardar.
    config (dict): Configuración que debe contener las claves 'rag' para nombrar el archivo y 'show_chunks' para determinar el número de chunks a mostrar.

    Returns:
    str: Nombre del archivo creado.
    """

    show_chunks = config.get('show_chunks', 3) 
    if len(chunks) < show_chunks:
        raise ValueError(f"La lista de chunks debe contener al menos {show_chunks} elementos.")

    df_chunks = pd.DataFrame(chunks[-show_chunks:])

    file_name = f"chunks_{config['rag']}.xlsx"

    df_chunks.to_excel(file_name, index=False, engine="openpyxl")

    return 
#Funcion para crear chunk mediante recursividad de caracteres y aplica limpieza para mejorar los embeddings:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
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


# Función para dividir texto en oraciones
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


# Función para combinar oraciones
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

        # Añadir oraciones previas
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        # Añadir oración actual
        combined_sentence += sentences[i]['sentence']

        # Añadir oraciones posteriores
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        # Guardar la oración combinada en el dict actual
        sentences[i]['combined_sentence'] = combined_sentence.strip()

    return sentences

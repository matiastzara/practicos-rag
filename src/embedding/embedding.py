import re  
from typing import List, Dict  
from sklearn.metrics.pairwise import cosine_similarity  
from langchain_huggingface import HuggingFaceEmbeddings 


def calculate_cosine_distances(sentences: List[Dict[str, str]], model_name: str) -> List[float]:
    """
    Calcula las distancias coseno entre embeddings de oraciones combinadas.

    Args:
        sentences (List[Dict[str, Any]]): Lista de oraciones con embeddings combinados.
        model_name (str): Nombre del modelo de embeddings.

    Returns:
        List[float]: Distancias coseno entre embeddings consecutivos.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embedding_model.embed_documents([sentence['combined_sentence'] for sentence in sentences])

    for i, sentence in enumerate(sentences):
        sentence['embedding'] = embeddings[i]

    distances = list()
    for i in range(len(sentences) - 1):
        sim = cosine_similarity([sentences[i]['embedding']], [sentences[i + 1]['embedding']])[0][0]
        distances.append(1 - sim)

    return distances

def split_into_chunks(sentences: List[Dict[str, str]], distances: List[float], threshold: float) -> List[str]:
    """
    Divide el texto en fragmentos basado en la distancia coseno entre oraciones.

    Args:
        sentences (List[Dict[str, str]]): Lista de oraciones.
        distances (List[float]): Distancias entre oraciones consecutivas.
        threshold (float): Umbral para decidir la separación de fragmentos.

    Returns:
        List[str]: Lista de fragmentos de texto.
    """
    chunks = list()
    start_index = 0

    for i, distance in enumerate(distances):
        if distance > threshold:
            chunk = ' '.join(sentence['sentence'] for sentence in sentences[start_index:i + 1])
            chunks.append(chunk)
            start_index = i + 1

    if start_index < len(sentences):
        chunk = ' '.join(sentence['sentence'] for sentence in sentences[start_index:])
        chunks.append(chunk)

    return chunks

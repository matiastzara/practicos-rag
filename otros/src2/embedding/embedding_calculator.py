from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_distances(sentences, model_name):
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

    distances = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity([sentences[i]['embedding']], [sentences[i + 1]['embedding']])[0][0]
        distances.append(1 - sim)

    return distances

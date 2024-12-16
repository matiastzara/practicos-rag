import re

def split_text_into_sentences(text: str):
    """
    Divide un texto en oraciones basado en '.', '?', y '!' y devuelve una lista de diccionarios.
    Args:
        text (str): El texto a dividir.
    Returns:
        List[Dict[str, str]]: Lista de diccionarios con 'sentence' y 'index'.
    """
    single_sentences_list = re.split(r'(?<=[.?!])\s+', text.strip())
    return [{'sentence': sentence, 'index': i} for i, sentence in enumerate(single_sentences_list)]


def combine_sentences(sentences, buffer_size=1):
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

        sentences[i]['combined_sentence'] = combined_sentence.strip()
    return sentences


def split_into_chunks(sentences, distances, threshold):
    """
    Divide el texto en fragmentos basado en la distancia coseno entre oraciones.
    Args:
        sentences (List[Dict[str, str]]): Lista de oraciones.
        distances (List[float]): Distancias entre oraciones consecutivas.
        threshold (float): Umbral para decidir la separación de fragmentos.
    Returns:
        List[str]: Lista de fragmentos de texto.
    """
    chunks = []
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

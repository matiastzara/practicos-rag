from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import torch
from langchain.embeddings.base import Embeddings

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "avsolatorio/NoInstruct-small-Embedding-v0", mode: str = "sentence"):
        """
        Clase personalizada para generar embeddings utilizando modelos de Hugging Face.

        :param model_name: Nombre del modelo de Hugging Face.
        :param mode: Modo para calcular embeddings ('query' o 'sentence').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.mode = mode
        self.dimension = self.model.config.hidden_size  # Obtén la dimensión desde la configuración del modelo

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de documentos.

        :param texts: Lista de cadenas de texto.
        :return: Lista de embeddings para los documentos.
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Genera un embedding para una consulta.

        :param text: Cadena de texto de la consulta.
        :return: Embedding para la consulta.
        """
        return self._get_embeddings([text])[0]

    def get_dimension(self) -> int:
        """
        Retorna la dimensión del vector generado por el modelo.

        :return: Dimensión del vector.
        """
        return self.dimension

    def _get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Genera embeddings utilizando el modelo predefinido.

        :param texts: Lista de cadenas de texto o una sola cadena.
        :return: Lista de embeddings.
        """
        self.model.eval()

        if isinstance(texts, str):
            texts = [texts]

        inp = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**inp)

        if self.mode == "query":
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:
            vectors = output.last_hidden_state[:, 0, :]  # CLS token

        return vectors.cpu().numpy().tolist()
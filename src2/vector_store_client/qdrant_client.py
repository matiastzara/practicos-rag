from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import RetrievalMode

def create_qdrant_store(model_name, chunks):
    """
    Crea y devuelve un QdrantVectorStore a partir de un modelo de embeddings y una lista de chunks de texto.

    Args:
        model_name (str): Nombre del modelo de embeddings.
        chunks (List[str]): Lista de fragmentos de texto.

    Returns:
        QdrantVectorStore: Objeto de almacenamiento Qdrant.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    documents = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]

    qdrant = QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        location=":memory:",  # Puedes cambiar la ubicación para persistencia
        collection_name="my_documents",
        retrieval_mode=RetrievalMode.DENSE,
    )

    return qdrant

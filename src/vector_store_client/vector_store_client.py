from typing import List, Dict  
from uuid import uuid4 
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore  
from qdrant_client.http.models import Distance, VectorParams 
from qdrant_client import QdrantClient 
from sentence_transformers import SentenceTransformer


def create_qdrant_store(model_name: str, chunks: List[Dict[str, str]]) -> QdrantVectorStore:
    """
    Crea y devuelve un QdrantVectorStore a partir de un modelo de embeddings y una lista de chunks de texto.

    Args:
        model_name (str): Nombre del modelo de embeddings.
        chunks (List[Dict[str, str]]): Lista de fragmentos de texto con metadatos.

    Returns:
        QdrantVectorStore: Objeto de almacenamiento Qdrant.
    """
    open_source_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    documents_for_qdrant = [
        Document(
            page_content=chunk["chunk_text"],
            metadata={
                "title": chunk["metadata"].get("title", ""),
                "subtitle": chunk["metadata"].get("subtitle", ""),
                "sub_subtitle": chunk["metadata"].get("sub_subtitle", "")
            }
        )
        for chunk in chunks
    ]

    qdrant = QdrantVectorStore.from_documents(
        documents_for_qdrant,
        embedding=open_source_embeddings,
        sparse_embedding=sparse_embeddings,
        location=":memory:",  
        collection_name="my_documents",
        retrieval_mode=RetrievalMode.HYBRID,
    )

    return qdrant

def create_qdrant_store_naive(model_name: str, chunks: List[str]) -> QdrantVectorStore:
    """
    Crea y devuelve un QdrantVectorStore a partir de un modelo de embeddings y una lista de chunks de texto de manera sencilla.

    Args:
        model_name (str): Nombre del modelo de embeddings.
        chunks (List[str]): Lista de fragmentos de texto.

    Returns:
        QdrantVectorStore: Objeto de almacenamiento Qdrant.
    """
    open_source_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    model = SentenceTransformer(model_name)
    embedding_dimension = model.get_sentence_embedding_dimension()
    storage_path =  f"/tmp/langchain_qdrant10"
    name = "naive_documents10"
    client = QdrantClient(path=storage_path)

    try:
        client.get_collection(name)
    except ValueError:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embedding_dimension, 
                                        distance=Distance.COSINE),
        )

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=open_source_embeddings,
    )

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    qdrant.add_documents(documents=chunks, ids=uuids)

    return qdrant
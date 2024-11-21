
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

def create_vector_store(client, collection_name, embedding_model, splits, dimension, update = False):
    try:
        # Verificar si la colección ya existe
        if client.collection_exists(collection_name):
            if not update:
                print(f"La colección '{collection_name}' ya existe.")
            else:
                print(f"La colección '{collection_name}' ya existe. Será eliminada para actualizar...")
                client.delete_collection(collection_name)  # Elimina la colección existente
                print(f"La colección '{collection_name}' ha sido eliminada.")
                # Crear la nueva colección
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension,  # Dimensiones de los vectores
                        distance=Distance.COSINE  # Métrica de distancia
                    )
                )
                print(f"La colección '{collection_name}' ha sido creada nuevamente.")
        else:
            print(f"No se encontró la colección '{collection_name}'. Creando una nueva...")
            # Crear la colección
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,  # Dimensiones de los vectores
                    distance=Distance.COSINE  # Métrica de distancia
                )
            )
            print(f"La colección '{collection_name}' ha sido creada exitosamente.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    from uuid import uuid4
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)   
    return vector_store
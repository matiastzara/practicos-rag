import os  
from dotenv import load_dotenv  
from typing import List, Dict  

from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser  
from langchain import hub  
from langchain.document_loaders import PyPDFLoader 
from langchain_qdrant import QdrantVectorStore 
from langchain_openai import ChatOpenAI 
import yaml  

from src.loaders.loaders import (load_pdf, split_pdf_documents)
from src.chunking.chunking import (
    load_pdf_all_documents,
    clean_text_and_exclude_sections,
    split_text_into_sentences,
    combine_sentences,
    extract_metadata,
    assign_metadata_to_chunks_with_context
)
from src.embedding.embedding import (calculate_cosine_distances, split_into_chunks)
from src.vector_store_client.vector_store_client import (create_qdrant_store, create_qdrant_store_naive)

def create_rag_chain(qdrant: QdrantVectorStore, llm: ChatOpenAI) -> QdrantVectorStore:
    """
    Crea y devuelve una cadena RAG (Retrieval-Augmented Generation) utilizando LangChain.

    Args:
        qdrant (QdrantVectorStore): Almacén de vectores configurado para recuperar documentos relevantes.
        llm (ChatOpenAI): Modelo de lenguaje configurado para generar texto.

    Returns:
        rag_chain, retriever: La cadena RAG configurada para generación y recuperación y el retriever asociado.
    """

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = qdrant.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def create_llm(model_name: str, temperature: float, openai_api_key: str) -> ChatOpenAI:
    """
    Crea un modelo LLM utilizando los parámetros proporcionados.

    Args:
        model_name (str): Nombre del modelo a utilizar.
        temperature (float): Grado de creatividad en las respuestas.
        openai_api_key (str): Clave de API de OpenAI para la autenticación.

    Returns:
        ChatOpenAI: Una instancia del modelo configurado.
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,  
        openai_api_key=openai_api_key
    )
    return llm

def initialize_rag(config: dict) -> object:
    """
    Inicializa los componentes de RAG (Retrieval-Augmented Generation) según el tipo especificado en la configuración.

    Args:
        config (dict): Diccionario de configuración que contiene las siguientes claves:
            - "rag" (str): Tipo de RAG a inicializar, puede ser "super" o "naive".
            - "model_name" (str): Nombre del modelo a utilizar.
            - "model" (str): Ruta o identificador del modelo de lenguaje a usar.
            - "temperature" (float): Parámetro de temperatura para el modelo de lenguaje.
            - "openai_api_key" (str): Clave de API para OpenAI.
            - "directory_path" (str, opcional): Ruta al directorio que contiene archivos PDF (requerido para RAG "super").
            - "buffer_size" (int, opcional): Número de oraciones a combinar en un segmento (requerido para RAG "super").
            - "threshold" (float, opcional): Umbral para dividir en segmentos basado en distancia coseno (requerido para RAG "super").
            - "max_previous_chunks" (int, opcional): Número de segmentos previos a incluir como contexto (requerido para RAG "super").
            - "file_path" (str, opcional): Ruta a un archivo PDF único (requerido para RAG "naive").

    Returns:
        object: Objeto de cadena RAG inicializado según la configuración especificada.

    Raises:
        ValueError: Si la clave "rag" en la configuración no es "super" o "naive".
    """
    rag_type = config["rag"]
    model_name = config["model_name"]
    model = config["model"]
    temperature = config["temperature"]
    openai_api_key = config["openai_api_key"]

    if rag_type == "super":
        pdf_texts = load_pdf_all_documents(config["directory_path"])
        cleaned_text = clean_text_and_exclude_sections(" ".join(pdf_texts))
        sentences = split_text_into_sentences(cleaned_text)
        combined_sentences = combine_sentences(sentences, config["buffer_size"])
        distances = calculate_cosine_distances(combined_sentences, model_name)
        chunks = split_into_chunks(combined_sentences, distances, config["threshold"])
        annotated_chunks = assign_metadata_to_chunks_with_context(chunks, config["max_previous_chunks"])
        qdrant_store = create_qdrant_store(model_name, annotated_chunks)
        llm = create_llm(model, temperature, openai_api_key)
        rag_chain, retriever  = create_rag_chain(qdrant_store, llm)
        return rag_chain, retriever, annotated_chunks

    elif rag_type == "naive":
        docs = load_pdf(config["file_path"])
        naive_chunks = split_pdf_documents(docs)
        naive_qdrant = create_qdrant_store_naive(model_name, naive_chunks)
        llm = create_llm(model, temperature, openai_api_key)
        rag_chain, retriever  = create_rag_chain(naive_qdrant, llm)
        return rag_chain, retriever,naive_chunks
    else:
        raise ValueError("El valor de 'rag' en la configuración no es válido. Debe ser 'super' o 'naive'.")

def load_config(file_path):
    """
    Carga un archivo de configuración en formato YAML y reemplaza las variables de entorno en los valores correspondientes.

    Args:
        file_path (str): Ruta al archivo YAML que contiene la configuración.

    Returns:
        dict: Diccionario con la configuración cargada. Las variables en formato ${VAR_NAME} serán reemplazadas
              por el valor correspondiente de las variables de entorno. Si una variable no está definida,
              se asignará el valor 'MissingEnvVar: VAR_NAME'.
    """
    load_dotenv()
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, f"MissingEnvVar: {env_var}")
        return config

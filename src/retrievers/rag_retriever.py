from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI

def create_rag_chain(model, openai_api_key, qdrant):
    """
    Crea una cadena RAG (Retrieval-Augmented Generation) usando LangChain.
    
    :param model: Nombre del modelo OpenAI (str).
    :param openai_api_key: Clave de la API de OpenAI (str).
    :param qdrant: Objeto QdrantVectorStore configurado como un retriever.
    :return: Objeto rag_chain.
    """
    # Configurar el modelo OpenAI
    llm = ChatOpenAI(
        model=model,
        temperature=0.7,  # Ajusta la creatividad según sea necesario
        openai_api_key=openai_api_key
    )

    # Descargar y configurar el prompt desde LangChain Hub
    prompt = hub.pull("rlm/rag-prompt")

    # Función para formatear los documentos
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Configurar el retriever desde Qdrant
    #retriever = qdrant.as_retriever()
    retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Crear la cadena RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

import pickle
#from src.loaders.pdf_loader import load_pdf_all_documents, clean_text_and_exclude_sections
#from src.chunking.sentence_splitter import split_text_into_sentences, combine_sentences, split_into_chunks
#from src.embedding.embedding_calculator import calculate_cosine_distances
#from src.vector_store_client.qdrant_client import create_qdrant_store
from src.retrievers.rag_retriever import create_rag_chain

def initialize_data_pipeline(directory_path, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                             buffer_size=2, threshold=0.5, openai_api_key=None):
    """Inicializa el pipeline de datos y crea el modelo RAG."""
    #pdf_texts = load_pdf_all_documents(directory_path)
    #full_text = " ".join(pdf_texts)
    #cleaned_text = clean_text_and_exclude_sections(full_text)

    #sentences = split_text_into_sentences(cleaned_text)
    #combined_sentences = combine_sentences(sentences, buffer_size)
    #distances = calculate_cosine_distances(combined_sentences, model_name)
    #chunks = split_into_chunks(combined_sentences, distances, threshold)

    #qdrant_store = create_qdrant_store(model_name, chunks)
    with open('qdrant_store.pkl', 'rb') as f:
        qdrant_store = pickle.load(f)

    return create_rag_chain("gpt-3.5-turbo", openai_api_key, qdrant_store)

def handle_user_query(user_input, rag_chain):
    """Procesa la consulta del usuario y muestra la respuesta en la interfaz."""
    import streamlit as st

    if user_input.lower() == "adios":
        st.stop()

    st.chat_message("user").write(user_input)

    with st.spinner("Procesando..."):
        try:
            response = rag_chain.invoke(user_input).strip()
        except Exception as e:
            response = f"Hubo un error al procesar tu consulta: {e}"

    st.chat_message("assistant").write(response)

    # Actualizar historial de chat
    st.session_state["chat_history"].append({"user": user_input, "assistant": response})
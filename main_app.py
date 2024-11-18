import os
from src.loaders.pdf_loader import load_pdf_all_documents, clean_text_and_exclude_sections
from src.chunking.sentence_splitter import split_text_into_sentences, combine_sentences, split_into_chunks
from src.embedding.embedding_calculator import calculate_cosine_distances
from src.vector_store_client.qdrant_client import create_qdrant_store
from src.retrievers.rag_retriever import create_rag_chain
import pickle

# Parámetros
directory_path = "../practicos-rag/data/usa/"
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
buffer_size = 2
threshold = 0.5
openai_api_key = os.getenv("OPENAI_API_KEY")

# Pipeline
pdf_texts = load_pdf_all_documents(directory_path)
full_text = " ".join(pdf_texts)
cleaned_text = clean_text_and_exclude_sections(full_text)

sentences = split_text_into_sentences(cleaned_text)
combined_sentences = combine_sentences(sentences, buffer_size)
distances = calculate_cosine_distances(combined_sentences, model_name)
chunks = split_into_chunks(combined_sentences, distances, threshold)

qdrant_store = create_qdrant_store(model_name, chunks)
with open('qdrant_store.pkl', 'wb') as f:
    pickle.dump(qdrant_store, f)

rag_chain = create_rag_chain("gpt-3.5-turbo", openai_api_key, qdrant_store)

'''
question = "What are the mandatory data elements that must be submitted in ACE for FDA articles?"
response = rag_chain.invoke(question)
print("Respuesta:", response)
'''
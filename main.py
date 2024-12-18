import os
import logging
from src.chunking.chunking import (show_chunks_streamlit)
from src.retrievers.retrievers import (initialize_rag, load_config)
from src.evaluation.evaluation import (
    evaluate_and_save_results
)


def main():

    logging.basicConfig(level=logging.INFO)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Load configuration
    config = load_config('config.yaml')

    # Initialize RAG components
    rag_chain, retriever, chunks = initialize_rag(config)
    
    show_chunks_streamlit(chunks, config)

    # Check if evaluation is True in the config and execute accordingly
    if config.get('evaluation', False):  # Defaults to False if 'evaluation' is not in config
        evaluate_and_save_results(rag_chain, retriever, config)
        logging.info("Evaluation and saving of results completed.")
    else:
        logging.info("Evaluation is disabled; skipping evaluation and saving of results.")

    # Example question and response
    question = "What are the procedures for milk pasteurization according to the regulations mentioned in the document?"
    response = rag_chain.invoke(question)
    logging.info(f"Respuesta: {response}")

if __name__ == "__main__":
    main()
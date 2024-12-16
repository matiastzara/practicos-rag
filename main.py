import os
import logging
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
    rag_chain, retriever = initialize_rag(config)

    # Evaluate and save results
    df_raga = evaluate_and_save_results(rag_chain, retriever, config)

    # Example question and response
    question = "What are the procedures for milk pasteurization according to the regulations mentioned in the document?"
    response = rag_chain.invoke(question)
    logging.info(f"Respuesta: {response}")

if __name__ == "__main__":
    main()
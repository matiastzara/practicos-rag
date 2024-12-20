import pandas as pd
import os
import logging
from tqdm import tqdm
import random
import streamlit as st

from datasets import Dataset
from typing import List, Dict, Tuple, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from ragas import evaluate
from ragas.metrics import ( faithfulness, answer_relevancy,  context_recall, context_precision )
from src.retrievers.retrievers import ( create_llm )


def generate_factoid_qa_prompt():
    """
    Genera un prompt mejorado para la creación de preguntas y respuestas factuales basado en un contexto.

    Returns:
        ChatPromptTemplate: Un objeto de plantilla de prompt para generación de QA.
    """
    QA_generation_prompt = ChatPromptTemplate.from_template("""
    Your task is to generate a *factoid question* and its corresponding *answer* based on the given context.

    Here are the rules:
    1. The *factoid question* must be directly answerable with a specific and concise piece of factual information from the context.
    2. Avoid using phrases like "according to the passage" or "based on the context" in your question.
    3. The question should resemble the style of queries typically entered in a search engine, focusing on clarity and relevance.
    4. The context provided will have a maximum token limit of 400 to 500 tokens.                                                        

    Please provide your response in the following format:

    Output:::
    Factoid question: (Your factoid question here)
    Answer: (The answer to the factoid question here)

    Here is the context:

    Context: {context}

    Output:::
    """)
    return QA_generation_prompt


def question_chain(context: str, prompt: str, config: dict) -> str:
    """
    Genera una respuesta basada en un contexto y un modelo de lenguaje configurado.

    Args:
        context (str): Texto que proporciona el contexto para la generación de la respuesta.
        prompt (str): Plantilla del prompt que será formateada con el contexto proporcionado.
        config (dict): Configuración para el modelo de lenguaje, incluyendo el nombre del modelo,
                       la temperatura de generación y la clave de API de OpenAI.

    Returns:
        str: Respuesta generada por el modelo de lenguaje tras procesar el contexto y el prompt.
    """
    class SimplePassthrough:
        def __call__(self, inputs: dict) -> dict:
            return inputs  

    passthrough = SimplePassthrough() 
    passthrough_output = passthrough({"context": context})  

    prompt_output = prompt.format(**passthrough_output)
    
    llm = create_llm(config["model"], config["temperature"], config["openai_api_key"])

    llm_output = llm.invoke(prompt_output)
    
    parsed_output = StrOutputParser().parse(llm_output)
    
    return parsed_output


def process_multiple_docs(docs: List[object], prompt: str, config: Dict[str, str], num_samples: int = 15) -> List[str]:
    """
    Procesa múltiples documentos seleccionando una muestra aleatoria y generando preguntas basadas en el contexto.

    Args:
        docs (List[object]): Lista de objetos de documentos, donde cada documento tiene un atributo `page_content`.
        prompt (str): Plantilla del prompt que será utilizada para generar preguntas.
        config (Dict[str, str]): Configuración para el modelo de lenguaje, incluyendo el nombre del modelo,
                                 la temperatura de generación y la clave de API de OpenAI.
        num_samples (int, opcional): Número de documentos a seleccionar aleatoriamente para procesar. 
                                     Por defecto es 15.

    Returns:
        List[str]: Lista de preguntas generadas por el modelo de lenguaje para cada documento de la muestra.
    """
    sampled_docs = random.sample(docs, num_samples)
    sampled_docs_processed = [doc.page_content for doc in sampled_docs]
    
    questions = [
        question_chain(sampled_context, prompt, config)
        for sampled_context in tqdm(sampled_docs_processed, desc="Processing questions")
    ]
    
    return questions

def extract_questions_and_answers(data: List[object]) -> Tuple[List[str], List[str]]:
    """
    Extrae preguntas y respuestas factuales de una lista de mensajes.

    Args:
        data (List[object]): Lista de objetos de mensajes, cada uno con un atributo `content` que contiene texto.

    Returns:
        Tuple[List[str], List[str]]: Una tupla que contiene dos listas:
            - La primera lista contiene las preguntas factuales extraídas.
            - La segunda lista contiene las respuestas correspondientes.
    """
    questions = list()
    answers = list()
    for message in data:
        if hasattr(message, "content"):
            content = message.content
            if "Factoid question:" in content and "Answer:" in content:
                factoid_question = content.split("Factoid question:")[1].split("\nAnswer:")[0].strip()
                answer = content.split("Answer:")[1].strip()
                questions.append(factoid_question)
                answers.append(answer)
    return questions, answers


def evaluate_rag_pipeline(rag_chain: object, retriever: object, questions: List[str], ground_truths: List[List[str]]) -> pd.DataFrame:
    """
    Realiza la inferencia con un pipeline RAG, evalúa los resultados y devuelve un DataFrame con las métricas.

    Args:
        rag_chain: El modelo RAG para generar respuestas.
        retriever: El componente de recuperación para obtener contextos relevantes.
        questions (list): Lista de preguntas para realizar la inferencia.
        ground_truths (list): Lista de respuestas esperadas (ground truths) para evaluación.

    Returns:
        pandas.DataFrame: DataFrame con los resultados de la evaluación.
    """
    answers = list()
    contexts = list()

    for query in questions:

        answers.append(rag_chain.invoke(query))
        relevant_docs = retriever.invoke(query)
        contexts.append([doc.page_content for doc in relevant_docs])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "reference": [gt[0] for gt in ground_truths] 
    })


    result = evaluate(
        dataset=dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    return df

def evaluate_and_save_results(rag_chain: object, retriever: object, config: dict, use_existing_questions: bool = True,
    questions_file: Optional[str] = "data/evaluation_data.xlsx") -> pd.DataFrame:
    """
    Evalúa la tubería RAG con opciones para generar preguntas o usar un archivo existente.

    Args:
        rag_chain (object): Cadena RAG inicializada para generar respuestas.
        retriever (object): Mecanismo de recuperación para buscar información relevante.
        config (dict): Configuración con claves como "file_path", "num_samples", y "rag".
        use_existing_questions (bool): Si es True, usa un archivo de preguntas existentes.
        questions_file (Optional[str]): Ruta al archivo Excel que contiene preguntas y respuestas.

    Returns:
        pd.DataFrame: DataFrame con resultados de la evaluación.

    Raises:
        ValueError: Si no se puede generar o encontrar el archivo de preguntas cuando es necesario.
    """
    try:
        # Crear la carpeta "data" si no existe
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Cargar o generar preguntas
        if use_existing_questions:
            if not os.path.exists(questions_file):
                st.warning(f"Archivo de preguntas '{questions_file}' no encontrado. Generando nuevas preguntas.")
                use_existing_questions = False
            else:
                questions_df = pd.read_excel(questions_file)
                if questions_df.empty:
                    raise ValueError(f"El archivo '{questions_file}' está vacío. No se puede continuar.")
                questions = questions_df["question"].tolist()
                ground_truths = questions_df["answer"].tolist()
        else:
            # Generar preguntas usando el modelo
            loader = PyPDFLoader(config["file_path"])
            docs = loader.load()
            if not docs:
                raise ValueError("No se pudieron cargar documentos. Verifica el archivo y la configuración.")

            prompt = generate_factoid_qa_prompt()
            new_questions = process_multiple_docs(docs, prompt, config, config.get("num_samples", 15))

            if not new_questions:
                raise ValueError("No se generaron preguntas. Verifica los documentos y el pipeline.")

            questions, ground_truths = extract_questions_and_answers(new_questions)
            if not questions or not ground_truths:
                raise ValueError("La extracción de preguntas y respuestas falló.")

            # Crear y guardar el archivo de evaluación
            questions_df = pd.DataFrame({
                "question": questions,
                "answer": ground_truths
            })
            questions_df.to_excel(questions_file, index=False, engine="openpyxl")
            st.success(f"Archivo de evaluación creado: '{questions_file}'.")

        # Evaluar con las preguntas obtenidas
        df_raga = evaluate_rag_pipeline(rag_chain, retriever, questions, ground_truths)
        if df_raga.empty:
            raise ValueError("El DataFrame de resultados está vacío. Verifica el pipeline de evaluación.")

        # Guardar resultados
        results_file = f"results_{config['rag']}.xlsx"
        df_raga.to_excel(results_file, index=False, engine="openpyxl")
        logging.info(f"Resultados guardados en: {results_file}")
        st.success(f"Resultados de evaluación guardados en: '{results_file}'.")

        # Mostrar el promedio de las métricas principales
        metric_mapping = {
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy"
        }
        metrics = list(metric_mapping.keys())
        metric_means = df_raga[metrics].mean() * 100  # Convertir a porcentaje
        renamed_metrics = metric_means.rename(index=metric_mapping)
        renamed_metrics_df = renamed_metrics.to_frame(name="Promedio (%)")

        st.markdown("### Promedio de Métricas del Modelo Evaluado")
        st.dataframe(renamed_metrics_df.style.format("{:.2f}%"))  # Mostrar como porcentaje con 2 decimales

        # Registrar el nombre del modelo evaluado
        model_name = config.get("rag", "Modelo desconocido")
        logging.info(f"Modelo evaluado: {model_name}")

        # Cambiar el parámetro para evitar reutilización de preguntas
        config["use_existing_questions"] = False

        return df_raga

    except Exception as e:
        logging.error(f"Error durante la evaluación: {e}")
        st.error(f"Error durante la evaluación: {e}")
        raise

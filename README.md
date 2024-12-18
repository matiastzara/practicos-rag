# Construcción de un sistema Retrieval-Augmented Generation (E2E)

Este proyecto está diseñado para responder preguntas sobre la industria alimenticia utilizando documentos oficiales del gobierno de los Estados Unidos (USA). Es un proyecto RAG (Retrieval-Augmented Generation) que combina técnicas avanzadas como procesamiento de texto, embeddings y almacenamiento vectorial para extraer y generar respuestas precisas basadas en los datos proporcionados. Además, se encuentra desarrollado siguiendo una estructura escalable, optimizada para su implementación en entornos de producción.

## Descripción general:

**Carpetas principales:**

### Carpetas principales:

- **`background/`**: Contiene módulos para la configuración de fondo y herramientas auxiliares.
  - `bg_settings.py`: Configuración del fondo de la aplicación.
  - `file_upload.py`: Gestión de carga de archivos.

- **`data/`**: Directorio para almacenar datos relacionados con el proyecto. Se puede usar para entradas o salidas de procesamiento.

- **`otros/`**: Carpeta reservada para contenido adicional o complementario al proyecto.

- **`src/`**: Contiene el código fuente principal.
  - **`chunking/`**:
    - `sentence_splitter.py`: Divisor de texto en fragmentos manejables.
  - **`embedding/`**:
    - `embedding_calculator.py`: Calculador de embeddings basado en el modelo configurado.
  - **`loaders/`**:
    - `pdf_loader.py`: Cargador y procesador de archivos PDF.
  - **`retrievers/`**:
    - `rag_retriever.py`: Implementación de un sistema de recuperación para cadenas RAG.
  - **`vector_store_client/`**:
    - `qdrant_client.py`: Cliente para interactuar con el almacenamiento vectorial basado en Qdrant.
    - `vector_store_client.py`: Manejo general de operaciones con almacenamiento de vectores.



**Archivos principales:**
-	**`main_appv2.py`**: Archivo principal para ejecutar la aplicación.
-	**`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.

## Instrucciones para ejecución:
**Instalación**
1.	Clona este repositorio:
```bash
   git clone https://github.com/matiastzara/practicos-rag.git
```
2. Crea un entorno virtual en tu computador:
```bash
    python -m venv .venv
```
3. Activa el entorno virtual:
```bash
    # On Mac
    source .venv/bin/activate
    # On Windows
    .venv\Scripts\activate
```
4. Instala las dependencias:
```bash
    pip install -r requirements.txt
```
**Uso:**
1.	Ejecuta el código principal utilizando:
```bash
streamlit run main_appv2.py
```
Se abrirá una interfaz para que puedas interactuar con el modelo. Luego, puedes realizar tus preguntas en la parte inferior.

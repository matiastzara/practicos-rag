import re
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

class SemanticChunker:
    """
    This class is used to split a document into chunks based on the cosine distance
    between sentences, a.k.a semantic similarity.
    
    Attributes:
        document (str): The document to split into chunks.
        buffer_size (int): The number of sentences to include in the buffer.
        model_name (str): The name of the model to use for embedding.
        sentences (list): The sentences of the document.
        dense_embedding_model (HuggingFaceEmbeddings): The model to use for embedding.
    """
    
    def __init__(self, document, buffer_size=1, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.document = document
        self.buffer_size = buffer_size
        self.model_name = model_name
        self.sentences = self.split_into_sentences()
        self.dense_embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
    
    def split_into_sentences(self):
        """
        This method splits the document into sentences.

        Returns:
            list: A list of dictionaries, each containing a sentence and its index.
        """
        single_sentences_list = re.split(r'(?<=\.\s|[!?]\s)', self.document)
        print(f"{len(single_sentences_list)} sentences were found")
        return [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
    
    def combine_sentences(self):
        """
        This method combines sentences into chunks.

        Returns:
            list: A list of dictionaries, each containing a combined sentence and its index.
        """
        for i in range(len(self.sentences)):
            combined_sentence = ''
            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += self.sentences[j]['sentence'] + ' '
            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(self.sentences):
                    combined_sentence += self.sentences[j]['sentence']
            self.sentences[i]['combined_sentence'] = combined_sentence
        return self.sentences

    def calculate_cosine_distances(self):
        """
        This method calculates the cosine distances between sentences.

        Returns:
            list: A list of distances between sentences.
        """
        # Ensure combined sentences are created
        self.combine_sentences()
        
        # Use embed_documents instead of encode
        numpy_embeddings = self.dense_embedding_model.embed_documents(
            [x['combined_sentence'] for x in self.sentences]
        )
        
        for i, sentence in enumerate(self.sentences):
            sentence['combined_sentence_embedding'] = numpy_embeddings[i]
        
        distances = []
        for i in range(len(self.sentences) - 1):
            embedding_current = self.sentences[i]['combined_sentence_embedding']
            embedding_next = self.sentences[i + 1]['combined_sentence_embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distance = 1 - similarity
            distances.append(distance)
            self.sentences[i]['distance_to_next'] = distance
        return distances
    
    def split_into_chunks(self, threshold):
        """
        This method splits the document into chunks based on the cosine distance between sentences.

        Returns:
            list: A list of chunks.
        """
        self.combine_sentences()  # Ensure combined sentences are created
        distances = self.calculate_cosine_distances()
        indices_above_thresh = [i for i, distance in enumerate(distances) if distance > threshold]
        start_index = 0
        chunks = []
        
        for index in indices_above_thresh:
            end_index = index
            group = self.sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1
        
        if start_index < len(self.sentences):
            combined_text = ' '.join([d['sentence'] for d in self.sentences[start_index:]])
            chunks.append(combined_text)
        
        return chunks

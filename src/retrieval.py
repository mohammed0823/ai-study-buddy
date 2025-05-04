import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
import streamlit as st

# Handles document storage, chunking, embeddings, and FAISS index creation
class AIDocumentStore:
    def __init__(self, dataset_path, index_path, chunk_size=500):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.documents = []
        self.document_metadata = []
        self.index_path = index_path

    # Loads the dataset and splits abstracts into chunks
    def load_and_split(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        for _, row in df.iterrows():
            chunks = self.chunk_text(row['abstract'], self.chunk_size)
            for chunk in chunks:
                self.documents.append(chunk)
                self.document_metadata.append({
                    'title': row['title'],
                    'url': row['url']
                })

    # Splits a block of text into word-based chunks
    def chunk_text(self, text, size):
        words = text.split()
        return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

    # Embeds all loaded documents using OpenAI's embedding model
    def embed_documents(self):
        client = OpenAI(api_key=st.session_state.get("openai_api_key"))
        embeddings = []
        for doc in tqdm(self.documents, desc="Embedding documents"):
            response = client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings).astype("float32")

    # Builds and saves a FAISS index from embedded documents
    def build_index(self):
        self.load_and_split()
        embeddings = self.embed_documents()
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        return index

    # Loads an existing FAISS index from disk
    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "You may need to run `build_index()` first to generate it."
            )
        return faiss.read_index(self.index_path)

    # Returns metadata for all chunks
    def get_metadata(self):
        return self.document_metadata

    # Returns all stored document chunks
    def get_documents(self):
        return self.documents

# Embeds a user query into a vector using OpenAI's API
def embed_query(query):
    client = OpenAI(api_key=st.session_state.get("openai_api_key"))
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding).astype("float32")
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        
        # Return dummy vector if failure
        return np.zeros(1536).astype("float32")
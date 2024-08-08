import os
import unicodedata
import pickle
from typing import List, Optional

from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain.schema.document import Document


def get_bm25_retriever(
        chunks: List[Document], 
        source: List[str],
        k: Optional[int] = 5
) -> BM25Retriever:
    for _, chunk in chunks.items():
        current_pdf_name = os.path.splitext(os.path.basename(chunk[0].metadata['source']))[0]
        if current_pdf_name == source:
            retriever = BM25Retriever.from_documents(chunk)
            retriever.k = k
            return retriever
        

def save_retrievers(retrievers, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(retrievers, file)


def load_saved_retrievers(file_path):
    with open(file_path, 'rb') as file:
        retrievers = pickle.load(file)
    return retrievers
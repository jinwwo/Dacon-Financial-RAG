import os
import pickle
import unicodedata
from typing import Dict, List, Optional

from langchain.schema.document import Document
from langchain_community.document_transformers import LongContextReorder
from langchain_teddynote.retrievers import KiwiBM25Retriever


def get_kiwi_bm25_retriever(
        chunks: List[Document], 
        source: List[str],
        k: Optional[int] = 5
) -> KiwiBM25Retriever:
    for _, chunk in chunks.items():
        current_pdf_name = os.path.splitext(os.path.basename(chunk[0].metadata['source']))[0]
        if current_pdf_name == source:
            retriever = KiwiBM25Retriever.from_documents(chunk)
            retriever.k = k
            return retriever
        
        
def load_prompt(prompt_path: str) -> Dict[str, str]:
    with open(prompt_path, 'r', encoding='utf-8') as file:
        PROMPT_TEMPLATE = file.read()
    return PROMPT_TEMPLATE
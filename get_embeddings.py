from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings

_DEFAULT_MODEL_ID = "BAAI/bge-m3"


def get_embeddings(
    device: Optional[str] = 'cuda',
    model_name: Optional[str] = _DEFAULT_MODEL_ID
):
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=_DEFAULT_MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings
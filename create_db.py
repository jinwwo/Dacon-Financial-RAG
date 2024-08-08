from get_embeddings import get_embeddings

import argparse
import os
import sys
import shutil
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reset_path", default=None)
    parser.add_argument("--source_path", required=False)
    parser.add_argument("--db_path", required=False)

    args = parser.parse_args()

    if args.reset_path:
        clear_database(args.reset)

    if not args.source_path or not args.db_path:
        parser.error(
            "[--source_path] and [--db_path] are required unless [--reset] is specified."
        )

    chroma_path, source_path = args.db_path, args.source_path

    chunks = process_pdfs(source_path)

    for _, chunk in tqdm(chunks.items(), desc="Processing PDFs"):
        pdf_name = os.path.splitext(os.path.basename(chunk[0].metadata["source"]))[0]
        add_to_chroma(chunk, chroma_path, pdf_name)


def process_pdfs(source_path: str) -> Dict[str, List[Document]]:
    documents = load_documents(source_path)
    chunkenized_text = split_documents(documents)
    chunks = chunk_group_by_source(chunkenized_text)
    return chunks


def chunk_group_by_source(
    chunkenized_text: List[Document],
) -> Dict[str, List[Document]]:

    chunks = {}
    for chunk in chunkenized_text:
        source = chunk.metadata["source"]
        if source not in chunks:
            chunks[source] = []
        chunks[source].append(chunk)
    return chunks


def load_documents(source_path: str) -> List[Document]:
    document_loader = PyPDFDirectoryLoader(source_path)
    return document_loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = 500,
    chunk_overlap: Optional[int] = 50,
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(
        chunks: List[Document], chroma_base_path: str, pdf_name: str
) -> None:
    chroma_path = os.path.join(chroma_base_path, pdf_name)
    db = Chroma(persist_directory=chroma_path, embedding_function=get_embeddings(device='cpu'))

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def clear_database(chroma_path: str) -> None:
    print("Clearing database")
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    sys.exit("Database cleared.")


if __name__ == "__main__":
    main()
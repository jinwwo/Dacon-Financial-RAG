from get_embeddings import get_embeddings
from utils import get_bm25_retriever
from load_model import load_model
from create_db import load_documents, split_documents, chunk_group_by_source

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

DEFAULT_CONFIG_PATH = "./config.json"


def rag(data_path: str, source_path: str, chroma_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(data_path)
    model = load_model(quantization=False)
    retrievers = load_retrievers(df, source_path, chroma_path)
    prompt_template = load_config["PROMPT_TEMPLATE"]

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering Questions"):
        source = row["Source"]
        question = row["Question"]

        retriever = retrievers[source]
        contexts = retriever.invoke(question)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in contexts])

        prompt_template = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt_template.format(context=context_text, question=question)

        print(f"Question: {question}")
        response = model.invoke(prompt)
        print(f"Answer: {response}\n")

        results.append(
            {
                "Source": row["Source"],
                "Source_path": row["Source_path"],
                "Question": question,
                "Answer": response,
            }
        )

    return results


def get_db(pdf_filename: str, chroma_path: str) -> Chroma:
    db_path = os.path.join(chroma_path, pdf_filename)
    return Chroma(
        persist_directory=db_path, embedding_function=get_embeddings(device="cpu")
    )


def load_retrievers(
    df: pd.DataFrame, source_path: str, chroma_path: str
) -> Dict[str, EnsembleRetriever]:
    retrievers = {}

    documents = load_documents(source_path)
    chunkenized_text = split_documents(documents)
    chunks = chunk_group_by_source(chunkenized_text)
    sources = df["Source"].unique()

    for source in tqdm(sources, desc="Loading retrievers"):
        db = get_db(source, chroma_path)
        bm25_retriever = get_bm25_retriever(chunks, source)
        chroma_retriever = db.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 8}
        )
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )
        retrievers[source] = retriever

    return retrievers


def load_config(config_path: str) -> Dict[str, str]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--chroma_path", type=str, required=True)
    parser.add_argument("--submission", type=str, required=True)

    args = parser.parse_args()

    results = rag(args.data_path, args.source_path, args.chroma_path)

    submit = pd.read_csv("./dataset/sample_submission.csv")
    submit["Answer"] = [item["Answer"] for item in results]
    submit["Answer"] = submit["Answer"].fillna("데이콘")
    submit.to_csv(args.submission, encoding="UTF-8-sig", index=False)


if __name__ == "__main__":
    main()
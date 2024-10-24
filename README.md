[![image](https://github.com/user-attachments/assets/b66b2660-f140-45b3-a143-20d9b6e007e6)](https://dacon.io/competitions/official/236295/overview/description)

# RAG(Retrival Augmented Generation) Pipeline
This repository contains a RAG pipeline for the competition of Dacon [[Link](https://dacon.io/competitions/official/236295/overview/description)]

# Overview
The competition hosted on DACON involves developing a natural language processing (NLP) algorithm to improve the search and usability of central government financial data. Participants are tasked with creating a question-answering system using datasets like fiscal reports and budget documents. The aim is to make vast amounts of fiscal data more accessible to the public and experts. The competition is organized by the Korea Fiscal Information Service and the Ministry of Economy and Finance.
The dataset consists of PDFs containing various forms of fiscal information such as tables, text, and images. It includes both single-column data as well as 2-columns formats.

- The approach used an ensemble retriever combining [Kiwi](https://github.com/bab2min/Kiwi) + BM25 with ChromaDB for information retrieval.
For embeddings, the model used was BAAI/bge-m3
- For text generation, the base model employed was [ko-gemma-2-9b-it](https://huggingface.co/rtzr/ko-gemma-2-9b-it).

This method aimed to optimize both retrieval and text generation processes for working with fiscal document datasets.

# Usage
### Create vector DB for retrieval
```bash
python create_db.py --source_path "ROOT_PATH_OF_SOURCES" --db_path "PATH_TO_SAVE_DB"
```

### Inference
```bash
python inference.py --data_path "PATH_OF_TEST_CSV" --source_path "ROOT_PATH_OF_SOURCES" --chroma_path "PATH_TO_SAVE_DB" --submission "NAME_OF_RESULT"
```

# Features
### parsing.py
1. The existing parsing library simply reads documents, so it cannot utilize the table information from the data. Therefore, Camelot is used to convert the tables into Markdown format, and text extraction is done via pdfplumber.
2. Some documents in the dataset had a 2-column (2 up layout) format. To handle this, an algorithm was implemented to parse the document from the top left to the bottom right.

### utils.py
The retriever can be specified for each document to allow for custom usage.

### load_model.py
Three models were tested, and the one with the best performance was selected for use:

- [rtzr/ko-gemma-2-9b-it](https://huggingface.co/rtzr/ko-gemma-2-9b-it) **(selected)**
- [maum-ai--Llama-3-MAAL-8B-Instruct-v0.1](https://huggingface.co/maum-ai/Llama-3-MAAL-8B-Instruct-v0.1)
- [LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)

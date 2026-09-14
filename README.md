# LoreMaster AI Capstone

LoreMaster is a local retrieval-augmented generation (RAG) assistant for questions about J.R.R. Tolkien's legendarium. It retrieves relevant passages from a controlled Tolkien lore collection, reranks them, and supplies the strongest records to a locally running Llama 3.2 model.

The project was created as an educational AI capstone to compare a source-grounded RAG assistant with the answers produced by a general-purpose local language model.

## How it works

1. `import_lore.py` downloads selected Tolkien Gateway articles and removes unwanted webpage sections.
2. `build_vector_db.py` divides the collected lore into chunks, creates embeddings, and stores them in Chroma.
3. `ask_loremaster.py` retrieves 20 candidate chunks for a question.
4. A cross-encoder reranks the candidates and selects the six strongest results.
5. Llama 3.2 answers using only the retrieved lore records.

## Technology

- Python
- LangChain
- Chroma
- Ollama with Llama 3.2
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker

## Project structure

```text
Loremaster-ai-capstone/
|-- data/raw/                 # Collected lore records
|-- src/
|   |-- import_lore.py        # Collect and clean source articles
|   |-- build_vector_db.py    # Build the Chroma vector database
|   |-- ask_loremaster.py     # Run the interactive assistant
|   |-- load_documents.py     # Demonstrate document loading/chunking
|   |-- test_banner.py        # Test the console banner
|   `-- test_embeddings.py    # Test embedding similarity
|-- .gitignore
|-- README.md
`-- requirements.txt
```

The generated `chroma_db/` directory and local Python environment are intentionally excluded from the repository.

## Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed locally
- Internet access during initial setup to download Python packages and models

## Installation

From the project root, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the local language model:

```powershell
ollama pull llama3.2
```

The embedding and reranking models will download automatically the first time they are used.

## Build the knowledge base

The repository already contains collected text records under `data/raw`. To retrieve the articles again from Tolkien Gateway, run:

```powershell
python src/import_lore.py
```

Build the local Chroma database:

```powershell
python src/build_vector_db.py
```

## Run LoreMaster

Ensure Ollama is running, then execute:

```powershell
python src/ask_loremaster.py
```

Enter a Tolkien question at the `You >` prompt. Enter `exit`, `quit`, or `bye` to close the program.

## Testing utilities

```powershell
python src/test_embeddings.py
python src/test_banner.py
python src/load_documents.py
```

## Scope and limitations

- Answers are limited by the articles and passages available in the local collection.
- Retrieval and reranking improve source relevance but do not guarantee a perfect answer.
- The prompt directs Llama 3.2 to use only retrieved records and avoid unsupported events, motives, causes, and chronology.
- The application is designed for local console use and does not include a graphical interface.

## Data source and attribution

Lore records in `data/raw` are derived from [Tolkien Gateway](https://tolkiengateway.net/), whose wiki content is available under the [Creative Commons Attribution-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-sa/4.0/). Each generated record preserves its article title, source URL, source site, and licence information.

Tolkien Gateway and this project are not affiliated with or endorsed by the Tolkien Estate or Middle-earth Enterprises. This repository is an educational demonstration.


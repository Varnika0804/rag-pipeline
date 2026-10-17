# RAG Pipeline: Question Answering over a PDF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Varnika0804/rag-pipeline/blob/main/RAG_Pipeline.ipynb)

A compact, end-to-end Retrieval-Augmented Generation (RAG) pipeline: load a PDF, split it into chunks, embed and index them, retrieve the most relevant passages for a question, answer with Gemini using only that context, and score the answer with an LLM judge.


## How it works

```
PDF -> clean & chunk -> embed -> vector store -> retrieve top-k -> Gemini answers -> LLM judge scores
```

| Stage | Tool |
|---|---|
| PDF loading | `PyPDFLoader` (LangChain) |
| Chunking | `RecursiveCharacterTextSplitter`, 500 characters with 50 overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Chroma (in-memory) |
| Generation and judging | Google Gemini via `langchain-google-genai` |

## Quick start

### Google Colab

1. Open the notebook with the badge above.
2. Add your key as a Colab secret named `GOOGLE_API_KEY` (key icon in the left sidebar) and enable notebook access.
3. Run all cells and upload a PDF when prompted.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your Google API key in .env
mkdir -p data                 # place your PDF at data/document.pdf
jupyter notebook RAG_Pipeline.ipynb
```

### Configuration

Settings live in one cell near the top of the notebook.

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | Chunking parameters |
| `TOP_K` | 10 | Passages given to the LLM |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `GEMINI_MODEL` (env var) | `gemini-3.5-flash` | Gemini model name. Change it if your account uses a different one. |

## Evaluation

Each answer gets three 0-5 scores from a second Gemini call:

- **Context relevance:** did retrieval find passages relevant to the question?
- **Groundedness:** is the answer supported by those passages?
- **Answer relevance:** does the answer address the question?

The notebook runs a small batch of questions (including one out-of-scope question that should get an "I don't know") and prints a table with averages.

Example from an earlier run, on the question "What is a guardrail metric?": the answer was a correct definition drawn from the retrieved passages, and all three scores were 5/5/5. The judge is the same model as the generator, so treat the scores as a rough signal.

## Limitations and next steps

- Fixed-size chunks can split a definition across two chunks.
- Pure vector search: a hybrid (BM25 + embeddings) retriever or a reranker would likely help.
- The evaluation set is tiny and the judge shares a model with the generator.
- The index is in memory only. Pass `persist_directory=` to Chroma to keep it.

## Repository layout

```
RAG_Pipeline.ipynb   the pipeline
requirements.txt     dependencies (unpinned; pin them after a working run with pip freeze)
.env.example         template for local secrets
.gitignore           keeps .env, PDFs, and local databases out of Git
README.md
```

## Notes

- **API keys:** never commit a key. The notebook reads it from Colab Secrets, a `.env` file, or a hidden prompt.
- **Documents:** the notebook was developed on a chapter of a published book. That PDF is not included, and `.gitignore` excludes `data/*.pdf` by default. Use a document you have the right to share, or add your own.
- **License:** add a `LICENSE` file before publishing (MIT is a common choice).

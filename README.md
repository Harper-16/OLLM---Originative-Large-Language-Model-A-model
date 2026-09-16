# OLLM: Originative Large Language Model

OLLM is a local, retrieval-based conversational AI prototype. It learns from
your documents and question-and-answer files, then finds the most relevant
knowledge for each prompt instead of sending data to a hosted API.

## Features

- Hybrid retrieval using lexical scoring, character n-grams, and transformer embeddings
- Paraphrase-aware similarity with `sentence-transformers/all-MiniLM-L6-v2`
- Question-and-answer memory with exact-match prioritization
- Support for plain text, Markdown, JSON, CSV, PDF, and DOCX files
- Persistent local indexes and knowledge metadata
- Graceful fallback to lexical search when the embedding model is unavailable

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- Internet access is needed the first time the transformer model is downloaded

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Train the local index from the included data:

```bash
python main.py train data
```

Start a conversation:

```bash
python main.py chat
```

The chat interface supports:

```text
/stats                 Show the number of indexed documents
/search <query>        Inspect the strongest matching documents
/exit                  Close the chat
```

You can also train on selected files or folders:

```bash
python main.py train data/conversation.txt data/smalltalk.txt
```

## Data Format

Question-and-answer text files use this format:

```text
User: What is a CPU?
Assistant: A CPU executes instructions and performs calculations.
```

Other supported files are indexed as documents. JSON files containing SQuAD-style
`data` are imported as answerable question-and-answer pairs.

## How Retrieval Works

During training, OLLM builds a local inverted index and normalized transformer
embeddings. At query time it combines:

1. Exact and token-level matching for precise terms and identifiers
2. Character n-gram similarity for small wording changes
3. Sentence embeddings for semantic matches and paraphrases

The best question-and-answer result returns its stored answer. Document results
return the strongest matching text chunk.

To select another sentence-transformer model:

```bash
export ORIN_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
python main.py train data
```

`ORIN_EMBEDDING_BATCH_SIZE` can be set to tune embedding batch size for the
available memory.

## Generated Files

Training writes local state to `models/`:

- `orin_retrieval.pkl` - generated retrieval index and embeddings
- `orin_memory.json` - normalized question-and-answer memory
- `knowledge_tree.json` - definition-question tree
- `orin_meta.json` - source fingerprints used to skip unchanged files

The generated pickle is ignored by Git because it can be rebuilt with the
training command.

## Project Status

This is an experimental local model and retrieval system. It is designed for
exploration and personal datasets, not as a production-grade language model.

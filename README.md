# Text Processing System

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage
```bash
python process_script.py
```

## What it does
- Loads Reuters news corpus
- Extracts entities and patterns
- Creates summaries using TF-IDF and TextRank
- Answers queries using document search

## Agentic System
The agent searches documents, extracts key information, and synthesizes answers to user queries. It processes multiple documents to answer research questions about financial markets, earnings, and economic topics.

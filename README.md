# Siddha Vaithiyam Assistant

AI-powered Tamil Siddha medicine assistant built with a **vectorless RAG** architecture.

This project takes a structured Siddha Q&A dataset, groups it into health lessons, builds a JSON knowledge tree, and uses an LLM to:

1. choose the right lesson
2. choose the most relevant Q&A nodes
3. generate a grounded Tamil answer using only those retrieved records

Unlike a typical embedding-based RAG pipeline, this project does **not** use a vector database.

---

## Overview

The system is designed for explainable retrieval over a small-to-medium structured knowledge base.

Instead of:

```text
documents -> chunks -> embeddings -> vector DB -> nearest neighbors -> answer
```

this project uses:

```text
dataset -> lessons -> JSON tree -> AI lesson routing -> exact node retrieval -> answer
```

This makes the system easier to debug, easier to explain in interviews, and more traceable for domain-sensitive answers.

---

## Features

- Tamil question answering over Siddha medicine data
- Structured lesson-based retrieval
- No embeddings or vector database
- Traceable answers with source node IDs
- Streamlit UI
- Groq-powered AI routing and answer generation
- GitHub and Streamlit Cloud friendly project layout

---

## Current Knowledge Base

The current working knowledge base contains:

- **19 lessons**
- **800 classified Q&A records**

The full raw dataset had 1082 rows.  
The current app uses the 800 cleaned and classified rows already included in `siddha_tree.json`.

---

## Project Flow

```text
Hugging Face dataset
 -> clean CSV
 -> lesson classification
 -> structured JSON tree
 -> vectorless RAG backend
 -> Streamlit app
```

Detailed flow:

```text
siddha_dataset.csv
 -> siddha_clean.csv
 -> siddha_with_lessons_clean.csv
 -> siddha_tree.json
 -> rag.py
 -> app.py
```

---

## Vectorless RAG Architecture

For each user question:

```text
User query
 -> AI selects relevant lesson(s)
 -> AI selects relevant Q&A node IDs
 -> app retrieves exact source records from siddha_tree.json
 -> AI generates final Tamil answer using only retrieved context
```

This avoids sending the whole dataset to the LLM and helps reduce hallucination.

---

## Project Structure

```text
.
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── INTERVIEW.md
├── siddha_tree.json
├── siddha_knowledge_base.md
├── siddha_dataset.csv
├── siddha_with_lessons_clean.csv
└── .env
```

Main files:

- `app.py` - Streamlit UI
- `rag.py` - vectorless RAG pipeline
- `siddha_tree.json` - structured knowledge tree
- `INTERVIEW.md` - full interview explanation of the project

---

## Tech Stack

- **Python**
- **Streamlit**
- **Groq API**
- **python-dotenv**
- **JSON / CSV**

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd sid
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```env
GRO_API_KEY=your_groq_api_key_here
```

The app also supports:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Run Locally

```bash
streamlit run app.py
```

If you are using the local venv directly:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Sample Queries

Try these Tamil queries in the app:

```text
முகப்பரு குணமாக என்ன மருந்து?
மலச்சிக்கல் தீர என்ன செய்யலாம்?
இருமல் குணமாக என்ன மருந்து?
```

---

## Example Retrieval Behavior

For the question:

```text
முகப்பரு குணமாக என்ன மருந்து?
```

The system may:

- select `Skin Care and Beauty`
- retrieve nodes like `Q596`, `Q597`, `Q598`, `Q600`
- generate a Tamil answer using only those source records

This makes the answer more explainable than a generic LLM response.

---

## Deployment

### Streamlit Cloud

1. Push this project to GitHub
2. Create a new Streamlit app
3. Set the entry point to:

```text
app.py
```

4. Add app secrets:

```toml
GRO_API_KEY = "your_groq_api_key_here"
```

The app will automatically load `siddha_tree.json` from the project root.

---

## Why This Project Is Interesting

- It uses **vectorless RAG** instead of embedding-based retrieval
- It is easy to explain in interviews
- It is traceable because answers come from exact source nodes
- It is small, practical, and domain-focused
- It shows end-to-end ownership: data cleaning, structuring, retrieval, UI, and deployment

---

## Limitations

- Current app uses 800 classified rows, not the full 1082-row dataset
- Some lesson labels are LLM-generated and can be refined manually
- The system depends on API availability from Groq
- This is an educational/demo assistant and not a replacement for professional medical advice

---

## Future Improvements

- classify the remaining 282 rows
- add lesson subtopics
- add evaluation for lesson and node selection quality
- add feedback collection from users
- add admin workflow for updating the tree
- add fallback search for uncategorized records

---

## Interview Support

A full interview explanation is included here:

- [`INTERVIEW.md`](./INTERVIEW.md)

That file explains:

- architecture
- design choices
- why vectorless RAG was used
- deployment
- likely interview questions and answers

---

## Safety Note

This project uses Siddha medicine Q&A data for retrieval and demo purposes.  
It should not be treated as professional medical diagnosis or emergency medical guidance.

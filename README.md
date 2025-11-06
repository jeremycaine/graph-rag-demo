# Graph RAG
Needs Python 3.12

## 0. Setup
Run `./setup.sh`

Copy `.env.sample` to `.env` and populate with Anthropic key and Astra keys.

## 1. Vanilla LLM
`python 1-vanilla-llm.py`

The State of the Union speech became the "Hello World" text for early GenAI experiements. This script asks 3 questions about Biden's speech in 2022.

The questions go direct to the Claude LLM and it responds with that is doens't have the context.

## 2. Comparing against Simple RAG approach
`python 2-simple-raf-compare.py`

This takes a long piece of text summarising about technology in 2024. It has three questions that are posed directly to the LLM, and then against the RAG context.

It doesn't always given an answer when just using the LLM. For the RAG response you can see the text in the `tech_report_2024.txt` that is had used in creating the response.

## 3. Vectorstore
`python 3-vectorstore-rag.py`

This version uses Astra DB as a vector store for documents (text, plus metadata) created from the ingestion chunk text. It puts the vectors in a collection called `acme_3_vectorstore`.

## 4. Vectorstore with Graph
`python 4-vectorstore-with-graph-rag.py`

Now graph metadata and topic detection is added in. Here a manually craftedset of keywords are grouped into topics which are added as metadata to the graph links associated with the documents that get inserted into the vector store.

The script compares the standard retrieval approach vs. a metadata one (the knowledge graph). For the Graph retrieval it first finds the docs using the standard retrieval approahc, then using the metadata. in the knowledge graph it finds related chunks, creating an expanded retrieved set of data. You see in the trace output many more more chunks being used.This is passed to the LLM for context in generation of response. This expanded set of relevant data means the LLM can generated a more accurate response.

## 5. Graph Metrics
`python 5-graph-metrics.py`

This is the same as `4. Vectorstore with Graph` but now with auto-discovery of topics and keywords for the knowledge graph. It also generates metrics about that show the improvement of using Graph based retrieval over Standard retrieval.




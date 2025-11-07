# Graph RAG
Needs Python 3.12

## 0. Setup
Run `./setup.sh`

Copy `.env.sample` to `.env` and populate with Anthropic key or Watsonx keys and Astra keys.

There are different input files to test:
- `./state_of_union_2022_clip.txt`
- `./state_of_union_2022_full.txt`
- `./tech_report_2024.txt`

Then you have different question sets you can adjust and use
```
# Questions
questions1 = [
    "How is AI used across different industries?",
    "What are all the climate technologies mentioned?",
    "What happened after quantum breakthroughs?" 
]

questions2 = [
    "What did Biden say about Ukraine?",
    "What did Biden say about inflation?",
    "What did Biden mention about COVID-19?"
]
```

## 1. Vanilla LLM
`python 1-vanilla-llm.py`

The State of the Union speech became the "Hello World" text for early GenAI experiements. This script asks 3 questions about Biden's speech in 2022.

The questions go direct to the Claude LLM and it responds with that is doens't have the context.

## 2. Comparing against Simple RAG approach
`python 2-simple-rag-compare.py`

This example works better with a longer piece of text. You could use a report text that summarisies technology in 2024. Or you could use the full text of the 2022 State of the Union speech. 

We then use three test questions that are posed directly to the LLM with no additional context. Then use a RAG search for additional contenxt (text) to ask with the question to the LLM.against the RAG context.

You get different results for Vanilla LLM and RAG context LLM. You can see that RAG is creating a better response - you will see text in the LLM response which is actual text that came from the input file.

## 3. Vectorstore
`python 3-vectorstore-rag.py`

This version uses Astra DB as a vector store for documents (text, plus metadata) created from the ingestion chunk text. It puts the vectors in a collection called `acme_3_vectorstore`.

## 4. Vectorstore with Graph
`python 4-vectorstore-with-graph-rag.py`

Now graph metadata and topic detection is added in. Here a manually craftedset of keywords are grouped into topics which are added as metadata to the graph links associated with the documents that get inserted into the vector store.

The script compares the standard retrieval approach vs. a metadata one (the knowledge graph). For the Graph retrieval it first finds the docs using the standard retrieval approahc, then using the metadata. in the knowledge graph it finds related chunks, creating an expanded retrieved set of data. You see in the trace output many more more chunks being used.This is passed to the LLM for context in generation of response. This expanded set of relevant data means the LLM can generated a more accurate response.

## 5. Graph Metrics
`python 5-metrics-and-graph-topics.py`

This is the same as `4. Vectorstore with Graph` but now with auto-discovery of topics and keywords for the knowledge graph. It also generates metrics about that show the improvement of using Graph based retrieval over Standard retrieval.




# Graph RAG

## Setup
Use Python 3.12

Run `./setup.sh`

Copy `.env.sample` to `.env` and populate with Anthropic key or Watsonx keys and Astra keys.

There are different input files to test:
- `./state_of_union_2022_clip.txt`
- `./state_of_union_2022_full.txt`
- `./tech_report_2024.txt`

Then you have different question sets you can use or change
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

## Main demo scripts
- [vanilla-llm.py](./vanilla-llm.py) simple use of Anthropic Claude LLM to answers questions
- [vanilla-vs-rag.py](./vanilla-vs-rag.py) use a RAG approach, holding chunks and documents in memory and compare answers to questions against vanilla LLM
- [wx-simple-rag-vs-graph.py](./wx-simple-rag-vs-graph-rag.py) use IBM watsonx.ai and its open source models; storing in a vector database, compare simple upload vs knowledge grpah upload

## IBM watsonx.ai setup

https://dataplatform.cloud.ibm.com/wx/home?context=wx 

Projects > New Project > 
    - Name: `graph-rag`
    - Select object storage: `my-object-storage`

Project `graph-rag`
- Manage > Service Integrations > Associate Service
    - `Watson Machine Learning-1f3` (find the instance of machine learning engine)
        - Plan: Essentials
        - Location: Dallas

- Manage > General
    Project id: `[get the project id]` (WATSONX_PROJECT_ID)
    
Get an API key (WATSONX_API_KEY)
- IBM Cloud - Manage IAM > API Key






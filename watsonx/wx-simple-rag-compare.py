import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from dotenv import load_dotenv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(override=True)

# Get credentials
# 1. Get credentials
wx_project_id = os.getenv("WX_PRODUCT_ID")
wx_api_key = os.getenv("WX_API_KEY")

credentials = Credentials(
    api_key=wx_api_key,
    url="https://us-south.ml.cloud.ibm.com"  # Change region if needed
)

# 2. Choose a foundation model and initialize Model
model_id = "ibm/granite-3-8b-instruct"  # Or any from the supported list

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=wx_project_id,
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 100,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0
    }
)

# get input file
input_file = os.getenv("INPUT_FILE")

# Verify they loaded
print("=" * 80)
print("RAG Demo: simple in-memory RAG")
print("=" * 80)
print(f"Input file to load: {input_file if input_file else 'NOT FOUND'}")

# Load the input file
print("\nLoading input file...")
with open(input_file, "r", encoding="utf-8") as f:
    speech_text = f.read()
print(f"Loaded {len(speech_text)} characters")

# Split text into chunks
print("\nSplitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_text(speech_text)
print(f"Created {len(chunks)} chunks")

# Create embeddings
# Initialize embedding model (using HuggingFace instead of OpenAI)
print("\nLoading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# using chunks directly for embedding, as we don't need full Document objects here with metadata
print("Creating embeddings for chunks...")
chunk_embeddings = embedding.embed_documents(chunks)
chunk_embeddings = np.array(chunk_embeddings)  # Convert to numpy array
print(f"Created embeddings with shape {chunk_embeddings.shape}")

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

# Run comparisons
for question in questions2:
    print("\n\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # ==================== VANILLA LLM ====================
    print("\nVANILLA LLM RESPONSE (No Context):")
    print("-" * 80)
    
    #  input and generate text
    input_text = question
    generated_response = model.generate_text(prompt=input_text)
    print(generated_response)
    
    # ==================== RAG-ENHANCED ====================
    print("\n\nRAG-ENHANCED RESPONSE (With Retrieved Context):")
    print("-" * 80)
    
    # Create query embedding
    query_embedding = embedding.embed_query(question)
    query_embedding = np.array(query_embedding).reshape(1, -1)  # Reshape for cosine_similarity
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top 3 chunks
    top_k = 3
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([chunks[idx] for idx in top_indices])
    
    # Create prompt with context
    prompt = f"""Answer this question.

Context:
{context}

Question: {question}

Answer based on the provided context:"""
    
    #  input and generate text
    generated_response = model.generate_text(prompt=prompt)
    print(generated_response)
    
    
print("\n" + "=" * 80)
print("End - RAG Demo: simple in-memory RAG")
print("=" * 80)
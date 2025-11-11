import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import anthropic
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(override=True)

# Get credentials
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
input_file = os.getenv("INPUT_FILE")

# Verify they loaded
print("=" * 80)
print("RAG Demo: simple in-memory RAG")
print("=" * 80)
print(f"Anthropic Key loaded: {anthropic_api_key[:20] if anthropic_api_key else 'NOT FOUND'}...")
print(f"Input file to load: {input_file if input_file else 'NOT FOUND'}")

# Get Anthropic client
client = anthropic.Anthropic(api_key=anthropic_api_key)

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
questions = [
    "How is AI used across different industries?",
    "What are all the climate technologies mentioned?",
    "What happened after quantum breakthroughs?" 
]

# Run comparisons
for question in questions:
    print("\n\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # ==================== VANILLA LLM ====================
    print("\nVANILLA LLM RESPONSE (No Context):")
    print("-" * 80)
    
    vanilla_message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Answer this question: {question}"
            }
        ]
    )
    
    print(vanilla_message.content[0].text)
    
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
    
    rag_message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    print(rag_message.content[0].text)
    
    # Show retrieved chunks
    print("\n\nRETRIEVED CONTEXT (Top 3 chunks):")
    print("-" * 80)
    for i, idx in enumerate(top_indices, 1):
        print(f"\nChunk {i} (Similarity: {similarities[idx]:.4f}):")
        chunk_preview = chunks[idx][:300] + "..." if len(chunks[idx]) > 300 else chunks[idx]
        print(chunk_preview)

print("\n" + "=" * 80)
print("End - RAG Demo: simple in-memory RAG")
print("=" * 80)
"""
RAG Demo: State of the Union Speech
Compares RAG-enhanced responses vs vanilla LLM responses using Anthropic's Claude API
"""

import os
import anthropic
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    """Simple RAG system using sentence embeddings for retrieval"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model for embeddings"""
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.embeddings = None
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def ingest_document(self, text: str):
        """Ingest a document by chunking and creating embeddings"""
        print("Chunking document...")
        self.chunks = self.chunk_text(text)
        print(f"Created {len(self.chunks)} chunks")
        
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(self.chunks)
        print("Document ingested successfully!")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant chunks for a query"""
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return chunks with their similarity scores
        results = [(self.chunks[idx], similarities[idx]) for idx in top_indices]
        return results


class RAGComparison:
    """Compare RAG vs Vanilla LLM responses"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.rag_system = SimpleRAG()
        self.model = "claude-sonnet-4-20250514"
        
    def load_state_of_union(self, filepath: str):
        """Load the State of the Union speech"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        self.rag_system.ingest_document(text)
        
    def query_vanilla(self, question: str) -> str:
        """Query Claude without RAG context"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"Answer this question about President Biden's State of the Union address: {question}"
                }
            ]
        )
        return message.content[0].text
    
    def query_with_rag(self, question: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Query Claude with RAG context"""
        # Retrieve relevant chunks
        relevant_chunks = self.rag_system.retrieve(question, top_k=3)
        
        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([chunk for chunk, score in relevant_chunks])
        
        # Create prompt with context
        prompt = f"""Based on the following excerpts from President Biden's State of the Union address, please answer the question.

Context:
{context}

Question: {question}

Answer based on the provided context:"""
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return message.content[0].text, relevant_chunks
    
    def compare(self, question: str):
        """Run comparison between vanilla and RAG approaches"""
        print("=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)
        
        print("\n📝 VANILLA LLM RESPONSE (No Context):")
        print("-" * 80)
        vanilla_response = self.query_vanilla(question)
        print(vanilla_response)
        
        print("\n\n🔍 RAG-ENHANCED RESPONSE (With Retrieved Context):")
        print("-" * 80)
        rag_response, retrieved_chunks = self.query_with_rag(question)
        print(rag_response)
        
#        print("\n\n📚 RETRIEVED CONTEXT (Top 3 chunks):")
#        print("-" * 80)
#        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
#            print(f"\nChunk {i} (Similarity: {score:.4f}):")
#            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        print("\n" + "=" * 80)





def main():
    """Main execution function"""
    print("🚀 RAG Demo: State of the Union Speech Analysis\n")
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set!")
        print("\nPlease set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Use local file
    filepath = "./state_of_the_union.txt"
    
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found!")
        print("Please make sure state_of_the_union.txt is in the current directory.")
        return
    
    # Initialize comparison system
    print("\n🔧 Initializing RAG system...")
    comparison = RAGComparison(api_key)
    
    # Load document
    print(f"📄 Loading document from {filepath}...")
    comparison.load_state_of_union(filepath)
    
    # Example questions
    questions = [
        "What did Biden say about Ukraine?",
        "What did Biden say about inflation?",
        "What did Biden mention about COVID-19?"
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING COMPARISONS")
    print("=" * 80)
    
    for question in questions:
        comparison.compare(question)
        print("\n")


if __name__ == "__main__":
    main()

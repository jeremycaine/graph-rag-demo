import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents.base import Document

# Load environment variables
load_dotenv(override=True)

# Get credentials
wx_project_id = os.getenv("WX_PRODUCT_ID")
wx_api_key = os.getenv("WX_API_KEY")
astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Verify they loaded
print("=" * 80)
print("RAG Demo: Graph-Enhanced Vector Store")
print("=" * 80)

credentials = Credentials(
    api_key=wx_api_key,
    url="https://us-south.ml.cloud.ibm.com"  # Change region if needed
)

# initialize foundation model
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

# Initialize embedding model
print("\nLoading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vector store
print("Connecting to AstraDB vector store...")
vectorstore = AstraDBVectorStore(
    collection_name="acme_4_graph",
    embedding=embedding,
    api_endpoint=astra_endpoint,
    token=astra_token,
)
print(f"Vector store connected")

# Clear existing data for clean reload
print("\nClearing existing data from vector store...")
try:
    vectorstore.clear()
    print("Vector store cleared")
except Exception as e:
    print(f"Could not clear vector store (might be empty): {e}")

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

print("\n🧩 Method 2: Clustering chunks by semantic similarity...")

def cluster_chunks_semantic(chunks, embedding_model, n_clusters=5):
    """Cluster chunks based on semantic embeddings"""
    if len(chunks) < n_clusters:
        n_clusters = max(2, len(chunks) // 2)
    
    print(f"   Creating embeddings for clustering...")
    chunk_embeddings = embedding_model.embed_documents(chunks)
    chunk_embeddings_array = np.array(chunk_embeddings)
    
    print(f"   Running K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(chunk_embeddings_array)
    
    # Create topic names from cluster centers
    topic_names = [f"topic_{i}" for i in range(n_clusters)]
    
    # Assign topics to chunks
    chunk_topics = []
    for i, label in enumerate(cluster_labels):
        chunk_topics.append([topic_names[label]])
    
    return chunk_topics, topic_names

chunk_topics, discovered_topics = cluster_chunks_semantic(chunks, embedding, n_clusters=5)
print(f"✅ Discovered {len(discovered_topics)} topic clusters")
print(f"   Topics: {discovered_topics}")

# Create documents with auto-discovered metadata
print("\n📚 Creating documents with graph relationships...")
docs = []

for i, chunk in enumerate(chunks):
    metadata = {
        "source": input_file,
        "chunk_id": i,
        "total_chunks": len(chunks),
        "graph_links": [],
        "topics": chunk_topics[i],  # From clustering
        "keywords": keywords_per_chunk[i] if i < len(keywords_per_chunk) else []
    }
    
    links_data = []
    
    # Sequential links
    if i > 0:
        links_data.append({"kind": "sequential", "tag": f"chunk_{i-1}", "direction": "bidir"})
    if i < len(chunks) - 1:
        links_data.append({"kind": "sequential", "tag": f"chunk_{i+1}", "direction": "bidir"})
    
    # Self-reference
    links_data.append({"kind": "identity", "tag": f"chunk_{i}", "direction": "bidir"})
    
    # Topic-based links (connect chunks in same cluster)
    for topic in chunk_topics[i]:
        links_data.append({"kind": "topic", "tag": topic, "direction": "bidir"})
    
    metadata["graph_links"] = links_data
    
    doc = Document(page_content=chunk, metadata=metadata)
    docs.append(doc)

print(f"✅ Created {len(docs)} documents with auto-discovered topics")

# Add documents to vector store
print(f"\n⬆️  Adding {len(docs)} documents to vector store...")
vectorstore.add_documents(docs)
print("✅ Documents added to vector store")

# Initialize LLM
print("\n🤖 Initializing Anthropic LLM...")
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=anthropic_api_key,
    temperature=0,
)

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
Here is the context: {context}

Based on the context above, answer this question: {question}
""")

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Questions to test
questions = [
    "How is AI used across different industries?",
    "What are all the climate technologies mentioned?",
    "What happened after quantum breakthroughs?" 
]

print("\n" + "=" * 80)
print("RUNNING COMPARISONS: Standard vs Metadata-Enhanced Retrieval")
print("=" * 80)

for question in questions:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # ==================== STANDARD VECTOR SEARCH ====================
    print("\n📊 STANDARD VECTOR SEARCH (similarity only):")
    print("-" * 80)
    
    standard_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    
    standard_chain = (
        {"context": standard_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    standard_response = standard_chain.invoke(question)
    print(standard_response.content)
    
    print("\n📚 Retrieved chunks (standard):")
    standard_docs = standard_retriever.invoke(question)
    for i, doc in enumerate(standard_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        topics = doc.metadata.get('topics', [])
        topics_str = f" [Topics: {', '.join(topics)}]" if topics else ""
        print(f"  Chunk {chunk_id}{topics_str}")
    
    # ==================== METADATA-ENHANCED SEARCH ====================
    print("\n\n🕸️  METADATA-ENHANCED SEARCH (similarity + graph metadata):")
    print("-" * 80)
    
    # First get initial results
    initial_docs = standard_retriever.invoke(question)
    
    # Expand using graph metadata
    expanded_chunk_ids = set()
    for doc in initial_docs:
        chunk_id = doc.metadata.get('chunk_id')
        expanded_chunk_ids.add(chunk_id)
        
        # Add sequential neighbors
        for link in doc.metadata.get('graph_links', []):
            if link['kind'] == 'sequential':
                neighbor_id = int(link['tag'].split('_')[1])
                expanded_chunk_ids.add(neighbor_id)
    
    # Retrieve expanded set
    expanded_docs = []
    for chunk_id in sorted(expanded_chunk_ids):
        # Search for document with this chunk_id
        results = vectorstore.similarity_search(
            question,
            k=50,  # Get more to find specific chunks
            filter={"chunk_id": chunk_id}
        )
        if results:
            expanded_docs.append(results[0])
    
    # Format and send to LLM
    expanded_context = format_docs(expanded_docs)
    expanded_response = llm.invoke(
        prompt.format_messages(context=expanded_context, question=question)
    )
    
    print(expanded_response.content)
    
    print("\n📚 Retrieved chunks (metadata-enhanced):")
    for i, doc in enumerate(expanded_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        topics = doc.metadata.get('topics', [])
        topics_str = f" [Topics: {', '.join(topics)}]" if topics else ""
        print(f"  Chunk {chunk_id}{topics_str}")
    
    print(f"\nℹ️  Standard retrieved {len(standard_docs)} chunks, Metadata-enhanced retrieved {len(expanded_docs)} chunks")

print("\n" + "=" * 80)
print("End - RAG Demo: Graph-Enhanced Vector Store")
print("=" * 80)
import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv(override=True)

# Get credentials
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
input_file = os.getenv("INPUT_FILE")

# Verify they loaded
print("=" * 80)
print("RAG Demo: Graph-Enhanced Vector Store")
print("=" * 80)
print(f"Anthropic Key loaded: {anthropic_api_key[:20] if anthropic_api_key else 'NOT FOUND'}...")
print(f"Astra Endpoint loaded: {astra_endpoint[:30] if astra_endpoint else 'NOT FOUND'}...")
print(f"Astra Token loaded: {astra_token[:20] if astra_token else 'NOT FOUND'}...")
print(f"Input file to load: {input_file if input_file else 'NOT FOUND'}")

# Initialize embedding model
print("\n🔧 Loading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vector store
print("🔧 Connecting to AstraDB vector store...")
vectorstore = AstraDBVectorStore(
    collection_name="acme_4_graph",
    embedding=embedding,
    api_endpoint=astra_endpoint,
    token=astra_token,
)
print(f"✅ Vector store connected")

# Clear existing data for clean reload
print("\n🗑️  Clearing existing data from vector store...")
try:
    vectorstore.clear()
    print("✅ Vector store cleared")
except Exception as e:
    print(f"⚠️  Could not clear vector store (might be empty): {e}")

# Load the input file
print("\n📄 Loading input file...")
with open(input_file, "r", encoding="utf-8") as f:
    speech_text = f.read()
print(f"✅ Loaded {len(speech_text)} characters")

# Split text into chunks
print("\n📝 Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_text(speech_text)
print(f"✅ Created {len(chunks)} chunks")

# Create documents with graph metadata
print("\n📚 Creating documents with graph relationships...")
docs = []
for i, chunk in enumerate(chunks):
    # Create metadata with graph information (as JSON-serializable dict)
    metadata = {
        "source": input_file,
        "chunk_id": i,
        "total_chunks": len(chunks),
        "graph_links": []  # Store links as serializable dicts
    }
    
    # Build graph links as JSON-serializable dictionaries
    links_data = []
    
    # 1. Sequential links: Link to previous and next chunks
    if i > 0:
        links_data.append({"kind": "sequential", "tag": f"chunk_{i-1}", "direction": "bidir"})
    if i < len(chunks) - 1:
        links_data.append({"kind": "sequential", "tag": f"chunk_{i+1}", "direction": "bidir"})
    
    # 2. Self-reference link using chunk ID
    links_data.append({"kind": "identity", "tag": f"chunk_{i}", "direction": "bidir"})
    
    # 3. Topic-based links (simple keyword matching for demo)
    chunk_lower = chunk.lower()
    topics = []

    # Artificial Intelligence
    if "ai" in chunk_lower or "artificial intelligence" in chunk_lower or "machine learning" in chunk_lower or "neural" in chunk_lower:
        topics.append("artificial_intelligence")
        links_data.append({"kind": "topic", "tag": "artificial_intelligence", "direction": "bidir"})
    
    # Quantum Computing
    if "quantum" in chunk_lower or "qubit" in chunk_lower:
        topics.append("quantum_computing")
        links_data.append({"kind": "topic", "tag": "quantum_computing", "direction": "bidir"})
    
    # Climate Technology
    if "climate" in chunk_lower or "renewable" in chunk_lower or "carbon" in chunk_lower or "solar" in chunk_lower or "wind" in chunk_lower or "battery" in chunk_lower:
        topics.append("climate_technology")
        links_data.append({"kind": "topic", "tag": "climate_technology", "direction": "bidir"})
    
    # Biotechnology
    if "biotech" in chunk_lower or "gene" in chunk_lower or "crispr" in chunk_lower or "healthcare" in chunk_lower or "medical" in chunk_lower or "drug" in chunk_lower:
        topics.append("biotechnology")
        links_data.append({"kind": "topic", "tag": "biotechnology", "direction": "bidir"})
    
    # Space Exploration
    if "space" in chunk_lower or "mars" in chunk_lower or "satellite" in chunk_lower or "spacex" in chunk_lower or "asteroid" in chunk_lower:
        topics.append("space_exploration")
        links_data.append({"kind": "topic", "tag": "space_exploration", "direction": "bidir"})
    
    # Cybersecurity
    if "cyber" in chunk_lower or "security" in chunk_lower or "encryption" in chunk_lower or "privacy" in chunk_lower or "breach" in chunk_lower:
        topics.append("cybersecurity")
        links_data.append({"kind": "topic", "tag": "cybersecurity", "direction": "bidir"})
    
    # Robotics
    if "robot" in chunk_lower or "automation" in chunk_lower or "drone" in chunk_lower or "autonomous" in chunk_lower:
        topics.append("robotics")
        links_data.append({"kind": "topic", "tag": "robotics", "direction": "bidir"})
    
    # Financial Technology
    if "crypto" in chunk_lower or "blockchain" in chunk_lower or "defi" in chunk_lower or "fintech" in chunk_lower or "digital currency" in chunk_lower:
        topics.append("financial_technology")
        links_data.append({"kind": "topic", "tag": "financial_technology", "direction": "bidir"})
    
    # Education Technology
    if "education" in chunk_lower or "learning" in chunk_lower or "virtual reality" in chunk_lower or "remote work" in chunk_lower or "training" in chunk_lower:
        topics.append("education_technology")
        links_data.append({"kind": "topic", "tag": "education_technology", "direction": "bidir"})
    
    # Smart Cities
    if "smart city" in chunk_lower or "urban" in chunk_lower or "transportation" in chunk_lower or "traffic" in chunk_lower:
        topics.append("smart_cities")
        links_data.append({"kind": "topic", "tag": "smart_cities", "direction": "bidir"})
     
    # Store graph links and topics in metadata
    metadata["graph_links"] = links_data
    metadata["topics"] = topics
    
    doc = Document(
        page_content=chunk,
        metadata=metadata
    )
    
    docs.append(doc)

print(f"✅ Created {len(docs)} documents with graph relationships")

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
import os
# Suppress tokenizers parallelism warning - needs to be upfront before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager

# Load environment variables
load_dotenv(override=True)

# Get credentials
wx_project_id = os.getenv("WX_PROJECT_ID")
wx_api_key = os.getenv("WX_API_KEY")
astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Verify they loaded
print("=" * 80)
print("RAG Demo: Graph-Enhanced Vector Store with KeybertLinkExtractor")
print("=" * 80)

credentials = Credentials(
    api_key=wx_api_key,
    url="https://us-south.ml.cloud.ibm.com"  # Change region if needed
)

# Initialize foundation model for generation
print("\nInitializing LLM...")
model_id = "ibm/granite-3-8b-instruct"

llm = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=wx_project_id,
    params={
        "decoding_method": "greedy",
        "max_new_tokens": 500,
        "min_new_tokens": 50,
        "repetition_penalty": 1.1,
        "temperature": 0.1,
    }
)

# Get input file
input_file = os.getenv("INPUT_FILE")

# Initialize embedding model
print("Loading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vector store
print("Connecting to AstraDB vector store...")
vectorstore = AstraDBVectorStore(
    collection_name="acme_graphrag_keybert",
    embedding=embedding,
    api_endpoint=astra_endpoint,
    token=astra_token,
)
print(f"Vector store connected")

# Clear existing data for clean reload
print("Clearing existing data from vector store...")
try:
    vectorstore.clear()
    print("Vector store cleared")
except Exception as e:
    print(f"Could not clear vector store (might be empty): {e}")

# Load the input file
print("Loading input file...")
with open(input_file, "r", encoding="utf-8") as f:
    speech_text = f.read()
print(f"Loaded {len(speech_text)} characters")

# Split text into chunks
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_text(speech_text)
print(f"Created {len(chunks)} chunks")

# Convert chunks to documents (initial docs without keywords)
print("\nCreating documents from chunks...")
temp_docs = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={
            "source": input_file,
            "chunk_id": i,
            "total_chunks": len(chunks),
        }
    )
    temp_docs.append(doc)

# Extract keywords using KeybertLinkExtractor
print("\nExtracting keywords with KeyBERT...")
keyword_extractor = KeybertLinkExtractor(kind="kw")

# Extract keywords but convert to simple lists
docs = []
for i, doc in enumerate(temp_docs):
    # Extract keywords as Link objects
    links = keyword_extractor.extract_one(doc)
    
    # Convert Link objects to a simple list of keyword strings
    keywords = [link.tag for link in links]
    
    # Create new document with keywords as a list in metadata
    new_doc = Document(
        page_content=doc.page_content,
        metadata={
            "source": input_file,
            "chunk_id": i,
            "total_chunks": len(chunks),
            "keywords": keywords  # Simple list, not Link objects
        }
    )
    docs.append(new_doc)
    
print(f"Extracted keywords for {len(docs)} documents")

# Show example of extracted keywords
print("\nExample document with keywords:")
if docs:
    example_doc = docs[0]
    print(f"Chunk {example_doc.metadata['chunk_id']}:")
    print(f"Content preview: {example_doc.page_content[:100]}...")
    keywords = example_doc.metadata.get('keywords', [])
    print(f"Extracted {len(keywords)} keywords: {keywords[:5]}")

# Add documents to vector store
print(f"\nAdding {len(docs)} documents to vector store...")
vectorstore.add_documents(docs)
print("Documents added to vector store")

# Create prompt template
prompt_template = """
Here is the context: {context}

Based on the context above, answer this question: {question}

Answer:"""

prompt = PromptTemplate.from_template(prompt_template)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to generate response using Watsonx AI
def generate_response(context, question):
    """Generate response using LLM"""
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.generate_text(prompt=formatted_prompt)
    return response

# Questions
questions = [
    "What did Biden say about Ukraine?",
    "What did Biden say about inflation?",
    "What did Biden mention about COVID-19?"
]

print("\n" + "=" * 80)
print("RUNNING COMPARISONS: Standard Vector Search vs GraphRAG")
print("=" * 80)

for question in questions:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # ==================== STANDARD VECTOR SEARCH ====================
    print("\nSTANDARD VECTOR SEARCH (similarity only):")
    print("-" * 80)
    
    standard_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    
    # Retrieve documents
    standard_docs = standard_retriever.invoke(question)
    
    # Format context and generate response
    standard_context = format_docs(standard_docs)
    standard_response = generate_response(standard_context, question)
    
    print(standard_response)
    
    print("\nRetrieved chunks (standard):")
    for i, doc in enumerate(standard_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        keywords = doc.metadata.get('keywords', [])[:3]
        keywords_str = f" [Keywords: {', '.join(keywords)}]" if keywords else ""
        print(f"  Chunk {chunk_id}{keywords_str}")
    
    # ==================== GRAPHRAG WITH GRAPH RETRIEVER ====================
    print("\n\nGRAPHRAG (similarity + graph traversal):")
    print("-" * 80)
    
    # Create GraphRetriever
    # Edge format is: (metadata_field, metadata_field_to_match)
    # We're saying: "follow documents that share keywords"
    graph_retriever = GraphRetriever(
        store=vectorstore,
        edges=[("keywords", "keywords")],  # Connect docs with shared keywords
        strategy=Eager(
            start_k=3,      # Initial vector search results
            k=12,           # Max total results after graph traversal
            max_depth=2     # Two hops in the graph
        )
    )
    
    # Retrieve documents using graph traversal
    graph_docs = graph_retriever.invoke(question)
    
    # Format context and generate response
    graph_context = format_docs(graph_docs)
    graph_response = generate_response(graph_context, question)
    
    print(graph_response)
    
    print("\nRetrieved chunks (GraphRAG):")
    for i, doc in enumerate(graph_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        keywords = doc.metadata.get('keywords', [])[:3]
        keywords_str = f" [Keywords: {', '.join(keywords)}]" if keywords else ""
        print(f"  Chunk {chunk_id}{keywords_str}")
    
    print(f"\nStandard retrieved {len(standard_docs)} chunks, GraphRAG retrieved {len(graph_docs)} chunks")
    
    # Show which chunks were added by graph traversal
    standard_chunk_ids = {doc.metadata.get('chunk_id') for doc in standard_docs}
    graph_chunk_ids = {doc.metadata.get('chunk_id') for doc in graph_docs}
    added_by_graph = graph_chunk_ids - standard_chunk_ids
    if added_by_graph:
        print(f"Chunks added by graph traversal: {sorted(added_by_graph)}")
    else:
        print(f"No chunks added by graph traversal")

print("\n" + "=" * 80)
print("End - RAG Demo: Graph-Enhanced Vector Store with KeybertLinkExtractor")
print("=" * 80)
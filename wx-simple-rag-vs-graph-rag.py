import os

# Suppress tokenizers parallelism warning - needs to be upfront before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Assuming your environment file is named 'watsonx.env' and is located in the same directory as your script.
ENV_FILE_PATH = "watsonx.env"

from dotenv import load_dotenv

# watsonx.ai run-time libraries
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

from langchain_ibm import WatsonxEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager

# --- Load environment variables ---
load_dotenv(override=True)

# Get credentials
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_LLM_MODEL_ID = os.getenv("WATSONX_LLM_MODEL_ID")
WATSONX_EMBEDDING_MODEL_ID = os.getenv("WATSONX_EMBEDDING_MODEL_ID")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# Verify they loaded
print("=" * 80)
print("RAG Demo: Graph-Enhanced Vector Store with KeybertLinkExtractor")
print("=" * 80)

# This block creates the main credential object using the API key and URL.
# This object will be used to authenticate all subsequent API calls to watsonx.ai.
credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

# Initialize foundation model for generation
print("\nInitializing watsonx LLM...")

# Initialize the LLM ModelInference client
# This block sets up the client for the generative Large Language Model (LLM).
# It defines the model to use (WATSONX_LLM_MODEL_ID), the project ID,
# and generation parameters (llm_params) like setting temperature to 0.0 for deterministic output.
llm_params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0.0, # Setting temperature to 0.0 for deterministic RAG answers
}

watsonx_llm = ModelInference(
    model_id=WATSONX_LLM_MODEL_ID,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
    params=llm_params
)

# Get input file
input_file = os.getenv("INPUT_FILE")

# 1. Initialize watsonx.ai Embeddings
# The WatsonxEmbeddings class reads credentials and project_id from os.environ
embedding = WatsonxEmbeddings(
    model_id=WATSONX_EMBEDDING_MODEL_ID,
    project_id=WATSONX_PROJECT_ID,
    url=WATSONX_URL
)

# Initialize vector store
print("Connecting to AstraDB vector store...")
vectorstore = AstraDBVectorStore(
    collection_name="acme_graphrag_keybert",
    embedding=embedding,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
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

# Extract keywords and store as Link objects (standard format for graph extraction)
# Store only JSON-safe types in metadata before adding to the Vector Store.
docs = []
for i, doc in enumerate(temp_docs):
    # Extract keywords as Link objects
    links = keyword_extractor.extract_one(doc)
    
    # Convert Link objects to a simple list of keyword strings
    keywords = [link.tag for link in links]
    
    # Create new document with JSON-safe metadata
    new_doc = Document(
        page_content=doc.page_content,
        metadata={
            "source": input_file,
            "chunk_id": i,
            "total_chunks": len(chunks),
            "keywords": keywords,  # LIST of strings - SAFE
            # DO NOT include "links": links (List of Link objects) here, 
            # as Link objects are not JSON serializable by AstraDB/AstraPy.
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
    response = watsonx_llm.generate_text(prompt=formatted_prompt)
    return response

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

print("\n" + "=" * 80)
print("RUNNING COMPARISONS: Standard Vector Search vs GraphRAG")
print("=" * 80)

# --- START OF NEW/MODIFIED CODE BLOCK ---
# 1. Initialize variables to store the final results outside the loop scope
last_question = ""
standard_docs_final = []
graph_docs_final = []

for i, question in enumerate(questions2):
    # This index check determines if it is the last question in the list
    is_last_question = (i == len(questions2) - 1)
    
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

    # Store the results only for the last question
    if is_last_question:
        last_question = question
        standard_docs_final = standard_docs
        graph_docs_final = graph_docs
        print("\nStored final retrieval results for visualization.")

print("\n" + "=" * 80)
print("End - RAG Demo: Graph-Enhanced Vector Store with KeybertLinkExtractor")
print("=" * 80)

import networkx as nx
import webbrowser
from pyvis.network import Network
from langchain_core.documents import Document
from IPython.display import IFrame, display
import json # Import json to handle the options dictionary

def visualize_graph_rag_comparison_pyvis(question, standard_docs, graph_docs, filename="graph_rag_results.html"):
    """
    Generates an interactive pyvis network graph visualizing the chunks retrieved 
    by the standard search vs those linked by the GraphRAG approach, 
    and saves it to an HTML file using net.set_options() for the title/legend.
    """
    
    # 1. Setup Data Structures (Retrieval and Keyword Logic)
    standard_chunk_ids = {doc.metadata.get('chunk_id') for doc in standard_docs}
    chunk_data = {
        doc.metadata.get('chunk_id'): {
            'keywords': doc.metadata.get('keywords', []),
            'is_standard': doc.metadata.get('chunk_id') in standard_chunk_ids,
            'doc_content': doc.page_content 
        }
        for doc in graph_docs
    }

    # 2. Build the NetworkX Graph
    G_nx = nx.Graph()
    chunk_ids = list(chunk_data.keys())
    for chunk_id, data in chunk_data.items():
        G_nx.add_node(chunk_id, keywords=data['keywords'], is_standard=data['is_standard'])

    # Add edges based on shared keywords
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            id1 = chunk_ids[i]
            id2 = chunk_ids[j]
            keywords1 = set(chunk_data[id1]['keywords'])
            keywords2 = set(chunk_data[id2]['keywords'])
            shared_keywords = keywords1.intersection(keywords2)
            if shared_keywords:
                G_nx.add_edge(id1, id2, shared_keywords=", ".join(list(shared_keywords)[:3]))

    # 3. Initialize Pyvis Network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.force_atlas_2based()

    # 4. Transfer Nodes and Edges from NetworkX to Pyvis
    STANDARD_COLOR = '#39A7D8'
    GRAPH_ADDED_COLOR = '#FF6347'

    for node_id in G_nx.nodes():
        data = G_nx.nodes[node_id]
        is_standard = data['is_standard']
        color = STANDARD_COLOR if is_standard else GRAPH_ADDED_COLOR
        
        keywords_preview = ', '.join(data['keywords'][:5])
        content_preview = chunk_data[node_id]['doc_content'][:150].replace('\n', ' ')
        
        title_text = f"Source: {'Initial Search (k=3)' if is_standard else 'Graph Traversal Added'}\n"
        title_text += f"Keywords: {keywords_preview}...\n"
        title_text += f"Content: {content_preview}..."
        
        net.add_node(n_id=node_id, 
                     label=f"Chunk {node_id}",
                     color=color,
                     title=title_text,
                     size=30)

    for source, target, data in G_nx.edges(data=True):
        edge_title = f"Shared Keywords: {data.get('shared_keywords', 'N/A')}"
        net.add_edge(source, target, 
                     title=edge_title, 
                     color='lightgray', 
                     width=2)

    # 5. Create HTML Heading/Legend Content
    # Use standard HTML with inline styling for the legend block
    html_heading_content = f"""
    <div style="background: rgba(0, 0, 0, 0.8); padding: 20px; border-radius: 8px; color: white; border: 1px solid gray; position: absolute; top: 10px; left: 10px; z-index: 1000;">
        <h2 style="color: white; margin-top: 0;">GraphRAG Comparison: {question}</h2>
        <p><strong>Node Color Key:</strong></p>
        <p style="color: {STANDARD_COLOR};">■ Initial Vector Search Chunks (Start-k=3)</p>
        <p style="color: {GRAPH_ADDED_COLOR};">■ Chunks Added by Graph Traversal</p>
        <p><em>Hover over a chunk for details; hover over an edge to see shared keywords.</em></p>
    </div>
    """
    
    # 6. Inject HTML using set_options() for compatibility (Fixes TypeError)
    
    # Set the graph title/legend using the core vis.js 'html' option
    # Note: This is an undocumented workaround often required in pyvis
    options = {
        "html": html_heading_content,
        "physics": {
            "stabilization": False
        }
    }
    
    # Since net.set_options() expects a JSON string, we need to convert the dict
    net.set_options(json.dumps(options))

   # 7. Save and Display
    net.save_graph(filename)
    print(f"Interactive graph saved to {filename}")
    
    # Auto-open in browser
    webbrowser.open('file://' + os.path.realpath(filename))
    
    return filename
# --- EXAMPLE USAGE ---
# To test this function, you need to call it with actual data.
# Assuming you run your RAG loop, you would call it like this:
#

last_question = "What did Biden say about Ukraine?" 
output = visualize_graph_rag_comparison_pyvis(last_question, standard_docs_final, graph_docs_final)

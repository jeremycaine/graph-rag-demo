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
print("RAG Demo: with Vector Store")
print("=" * 80)
print(f"Anthropic Key loaded: {anthropic_api_key[:20] if anthropic_api_key else 'NOT FOUND'}...")
print(f"Astra Endpoint loaded: {astra_endpoint[:30] if astra_endpoint else 'NOT FOUND'}...")
print(f"Astra Token loaded: {astra_token[:20] if astra_token else 'NOT FOUND'}...")
print(f"Input file to load: {input_file if input_file else 'NOT FOUND'}")

# Initialize embedding model (using HuggingFace instead of OpenAI)
print("\nLoading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vector store
print("Connecting to AstraDB vector store...")
vectorstore = AstraDBVectorStore(
    collection_name="acme_3_vectorstore",
    embedding=embedding,
    api_endpoint=astra_endpoint,
    token=astra_token,
)
print(f"Vector store connected: {vectorstore}")

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

# Create documents
print("\nCreating documents...")
docs = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={
            "source": input_file,
            "chunk_id": i,
        }
    )
    docs.append(doc)

# Add documents to vector store
print(f"Adding {len(docs)} documents to vector store...")
vectorstore.add_documents(docs)
print("Documents added to vector store")

# Initialize LLM (using Anthropic instead of OpenAI)
print("Initializing Anthropic LLM...")
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

# Create retriever
print("Setting up retriever...")
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)

# Create RAG chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Questions to test
questions = [
    "How is AI used across different industries?",
    "What are all the climate technologies mentioned?",
    "What happened after quantum breakthroughs?" 
]

# Run comparisons
print("\n" + "=" * 80)
print("RUNNING RAG QUERIES")
print("=" * 80)

for question in questions:
    print("\n" + "-" * 80)
    print(f"QUESTION: {question}")
    print("-" * 80)
    
    # Invoke the chain
    response = chain.invoke(question)
    
    print("\nRAG RESPONSE (with vector store retrieval):")
    print(response.content)
    
    # Show retrieved chunks
    print("\nRETRIEVED CHUNKS:")
    retrieved_docs = retriever.invoke(question)
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"\nChunk {i} (from chunk_id {doc.metadata.get('chunk_id', 'N/A')}):")
        print(preview)

print("\n" + "=" * 80)
print("End - RAG Demo: with Vector Store")
print("=" * 80)
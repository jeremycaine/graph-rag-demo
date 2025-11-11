import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

load_dotenv(override=True)

# Get credentials
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
astra_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
input_file = os.getenv("INPUT_FILE")

print("=" * 80)
print("RAG Demo: Automatic Topic Discovery with Semantic Clustering and Metrics")
print("=" * 80)
print(f"Input file to load: {input_file if input_file else 'NOT FOUND'}")

# Initialize embedding model
print("\n🔧 Loading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load and chunk text
print("\n📄 Loading input file...")
with open(input_file, "r", encoding="utf-8") as f:
    speech_text = f.read()
print(f"✅ Loaded {len(speech_text)} characters")

print("\n📝 Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_text(speech_text)
print(f"✅ Created {len(chunks)} chunks")

# Method 1: TF-IDF Keyword Extraction
print("\n🔍 Method 1: Extracting keywords using TF-IDF...")

def extract_keywords_tfidf(chunks, top_n=3):
    """Extract top keywords from each chunk using TF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=50,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(chunks)
        feature_names = vectorizer.get_feature_names_out()
        
        chunk_keywords = []
        for i in range(len(chunks)):
            scores = tfidf_matrix[i].toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            chunk_keywords.append(keywords)
        
        return chunk_keywords
    except:
        # Fallback if too few chunks
        return [[] for _ in chunks]

keywords_per_chunk = extract_keywords_tfidf(chunks, top_n=3)
print(f"✅ Extracted keywords for {len(keywords_per_chunk)} chunks")
print(f"   Example keywords: {keywords_per_chunk[0] if keywords_per_chunk else 'none'}")

# Method 2: Semantic Clustering
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

# Show sample metadata
print("\n📋 Sample document metadata:")
sample_doc = docs[0]
print(f"   Chunk ID: {sample_doc.metadata['chunk_id']}")
print(f"   Topics: {sample_doc.metadata['topics']}")
print(f"   Keywords: {sample_doc.metadata['keywords']}")
print(f"   Graph links: {len(sample_doc.metadata['graph_links'])} links")

# Initialize vector store
print("\n🔧 Connecting to AstraDB vector store...")
vectorstore = AstraDBVectorStore(
    collection_name="acme_5_topics_metrics",
    embedding=embedding,
    api_endpoint=astra_endpoint,
    token=astra_token,
)

# Clear and load
print("\n🗑️  Clearing existing data...")
try:
    vectorstore.clear()
    print("✅ Cleared")
except Exception as e:
    print(f"⚠️  {e}")

print(f"\n⬆️  Adding {len(docs)} documents to vector store...")
vectorstore.add_documents(docs)
print("✅ Documents added")

# Initialize LLM
print("\n🤖 Initializing LLM...")
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=anthropic_api_key,
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
Here is the context: {context}

Based on the context above, answer this question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Test questions
questions = [
    "How is AI used across different industries?",
    "What are all the climate technologies mentioned?",
    "What happened after quantum breakthroughs?"
]

print("\n" + "=" * 80)
print("RUNNING COMPARISONS: Standard vs Metadata-Enhanced Retrieval")
print("=" * 80)

# Track metrics across all questions
all_metrics = []

for question in questions:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # ==================== STANDARD VECTOR SEARCH ====================
    print("\n📊 STANDARD VECTOR SEARCH (similarity only):")
    print("-" * 80)
    
    standard_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    standard_chain = (
        {"context": standard_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    standard_response = standard_chain.invoke(question)
    print(standard_response.content)
    
    print("\n📚 Retrieved chunks (standard):")
    standard_docs = standard_retriever.invoke(question)
    
    # Calculate standard metrics
    standard_chunk_ids = []
    standard_topics_count = 0
    standard_unique_topics = set()
    standard_unique_keywords = set()
    
    for i, doc in enumerate(standard_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        standard_chunk_ids.append(chunk_id)
        topics = doc.metadata.get('topics', [])
        keywords = doc.metadata.get('keywords', [])
        standard_topics_count += len(topics)
        standard_unique_topics.update(topics)
        standard_unique_keywords.update(keywords)
        
        topics_str = f" [Topics: {', '.join(topics)}]" if topics else ""
        keywords_str = f" [Keywords: {', '.join(keywords[:2])}]" if keywords else ""
        print(f"  Chunk {chunk_id}{topics_str}{keywords_str}")
    
    # Calculate context diversity (chunk ID spread)
    if len(standard_chunk_ids) > 1:
        standard_chunk_spread = max(standard_chunk_ids) - min(standard_chunk_ids)
    else:
        standard_chunk_spread = 0
    
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
    
    # Calculate enhanced metrics
    enhanced_chunk_ids = []
    enhanced_topics_count = 0
    enhanced_unique_topics = set()
    enhanced_unique_keywords = set()
    
    for i, doc in enumerate(expanded_docs, 1):
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        enhanced_chunk_ids.append(chunk_id)
        topics = doc.metadata.get('topics', [])
        keywords = doc.metadata.get('keywords', [])
        enhanced_topics_count += len(topics)
        enhanced_unique_topics.update(topics)
        enhanced_unique_keywords.update(keywords)
        
        topics_str = f" [Topics: {', '.join(topics)}]" if topics else ""
        keywords_str = f" [Keywords: {', '.join(keywords[:2])}]" if keywords else ""
        print(f"  Chunk {chunk_id}{topics_str}{keywords_str}")
    
    # Calculate context diversity (chunk ID spread)
    if len(enhanced_chunk_ids) > 1:
        enhanced_chunk_spread = max(enhanced_chunk_ids) - min(enhanced_chunk_ids)
    else:
        enhanced_chunk_spread = 0
    
    # Calculate character counts for context size
    standard_context_chars = sum(len(doc.page_content) for doc in standard_docs)
    enhanced_context_chars = sum(len(doc.page_content) for doc in expanded_docs)
    
    # Calculate answer lengths
    standard_answer_length = len(standard_response.content)
    enhanced_answer_length = len(expanded_response.content)
    
    # Store metrics for this question
    question_metrics = {
        "question": question,
        "chunks_retrieved_standard": len(standard_docs),
        "chunks_retrieved_enhanced": len(expanded_docs),
        "chunk_increase_pct": ((len(expanded_docs) - len(standard_docs)) / len(standard_docs) * 100) if len(standard_docs) > 0 else 0,
        "context_chars_standard": standard_context_chars,
        "context_chars_enhanced": enhanced_context_chars,
        "context_increase_pct": ((enhanced_context_chars - standard_context_chars) / standard_context_chars * 100) if standard_context_chars > 0 else 0,
        "unique_topics_standard": len(standard_unique_topics),
        "unique_topics_enhanced": len(enhanced_unique_topics),
        "topic_coverage_increase": len(enhanced_unique_topics) - len(standard_unique_topics),
        "unique_keywords_standard": len(standard_unique_keywords),
        "unique_keywords_enhanced": len(enhanced_unique_keywords),
        "keyword_coverage_increase": len(enhanced_unique_keywords) - len(standard_unique_keywords),
        "chunk_spread_standard": standard_chunk_spread,
        "chunk_spread_enhanced": enhanced_chunk_spread,
        "spread_increase": enhanced_chunk_spread - standard_chunk_spread,
        "answer_length_standard": standard_answer_length,
        "answer_length_enhanced": enhanced_answer_length,
        "answer_detail_increase_pct": ((enhanced_answer_length - standard_answer_length) / standard_answer_length * 100) if standard_answer_length > 0 else 0,
    }
    
    all_metrics.append(question_metrics)
    
    # Display metrics for this question
    print("\n" + "─" * 80)
    print("📊 METRICS COMPARISON:")
    print("─" * 80)
    print(f"Chunks Retrieved:         {question_metrics['chunks_retrieved_standard']} → {question_metrics['chunks_retrieved_enhanced']} ({question_metrics['chunk_increase_pct']:+.1f}%)")
    print(f"Context Size (chars):     {question_metrics['context_chars_standard']:,} → {question_metrics['context_chars_enhanced']:,} ({question_metrics['context_increase_pct']:+.1f}%)")
    print(f"Unique Topics Covered:    {question_metrics['unique_topics_standard']} → {question_metrics['unique_topics_enhanced']} ({question_metrics['topic_coverage_increase']:+d})")
    print(f"Unique Keywords Covered:  {question_metrics['unique_keywords_standard']} → {question_metrics['unique_keywords_enhanced']} ({question_metrics['keyword_coverage_increase']:+d})")
    print(f"Chunk Spread (diversity): {question_metrics['chunk_spread_standard']} → {question_metrics['chunk_spread_enhanced']} ({question_metrics['spread_increase']:+d} chunks)")
    print(f"Answer Length (chars):    {question_metrics['answer_length_standard']:,} → {question_metrics['answer_length_enhanced']:,} ({question_metrics['answer_detail_increase_pct']:+.1f}%)")

# Display aggregate metrics
print("\n" + "=" * 80)
print("📈 AGGREGATE METRICS ACROSS ALL QUESTIONS")
print("=" * 80)

avg_chunk_increase = sum(m['chunk_increase_pct'] for m in all_metrics) / len(all_metrics)
avg_context_increase = sum(m['context_increase_pct'] for m in all_metrics) / len(all_metrics)
avg_topic_increase = sum(m['topic_coverage_increase'] for m in all_metrics) / len(all_metrics)
avg_keyword_increase = sum(m['keyword_coverage_increase'] for m in all_metrics) / len(all_metrics)
avg_spread_increase = sum(m['spread_increase'] for m in all_metrics) / len(all_metrics)
avg_answer_increase = sum(m['answer_detail_increase_pct'] for m in all_metrics) / len(all_metrics)

print(f"\nAverage Chunk Retrieval Increase:    {avg_chunk_increase:+.1f}%")
print(f"Average Context Size Increase:       {avg_context_increase:+.1f}%")
print(f"Average Topic Coverage Increase:     {avg_topic_increase:+.1f} topics")
print(f"Average Keyword Coverage Increase:   {avg_keyword_increase:+.1f} keywords")
print(f"Average Chunk Spread Increase:       {avg_spread_increase:+.1f} chunks")
print(f"Average Answer Detail Increase:      {avg_answer_increase:+.1f}%")

# Summary verdict
print("\n" + "─" * 80)
print("💡 INTERPRETATION:")
print("─" * 80)
if avg_chunk_increase > 50:
    print("✅ Metadata-enhanced retrieval provides SIGNIFICANTLY more context")
elif avg_chunk_increase > 20:
    print("✅ Metadata-enhanced retrieval provides MODERATELY more context")
else:
    print("⚠️  Metadata-enhanced retrieval provides MINIMAL additional context")

if avg_topic_increase > 1:
    print(f"✅ Metadata-enhanced covers {avg_topic_increase:.1f} more topics on average")
elif avg_topic_increase > 0:
    print("✅ Metadata-enhanced covers slightly more topics")
else:
    print("⚠️  Similar topic coverage between approaches")

if avg_keyword_increase > 3:
    print(f"✅ Metadata-enhanced covers {avg_keyword_increase:.1f} more keywords on average")
elif avg_keyword_increase > 0:
    print(f"✅ Metadata-enhanced covers slightly more keywords ({avg_keyword_increase:.1f})")
else:
    print("⚠️  Similar keyword coverage between approaches")

if avg_spread_increase > 3:
    print(f"✅ Metadata-enhanced retrieves from {avg_spread_increase:.1f} more diverse locations")
elif avg_spread_increase > 1:
    print("✅ Metadata-enhanced retrieves from moderately more diverse locations")
else:
    print("⚠️  Similar contextual diversity between approaches")

if avg_answer_increase > 30:
    print(f"✅ Metadata-enhanced produces {avg_answer_increase:.1f}% more detailed answers")
elif avg_answer_increase > 10:
    print(f"✅ Metadata-enhanced produces moderately more detailed answers ({avg_answer_increase:.1f}%)")
else:
    print("⚠️  Similar answer detail between approaches")

print("\n" + "─" * 80)
print("🎯 TOPIC DISCOVERY QUALITY:")
print("─" * 80)
print(f"Topics auto-discovered: {len(discovered_topics)}")
print(f"Average keywords per chunk: {sum(len(k) for k in keywords_per_chunk) / len(keywords_per_chunk):.1f}")
print(f"✅ All topics discovered automatically - no hard-coded keywords needed!")

print("\n" + "=" * 80)
print("✅ Demo Complete - Topics Auto-Discovered and Metrics Calculated")
print("=" * 80)
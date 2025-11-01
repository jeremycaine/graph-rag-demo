# Graph Metadata for Plain Text vs PDF Documents

## Do You Need ingestion processing from Unstructured.io?

**Short answer: No, not for plain text.**

Unstructured is specifically designed to extract hierarchical structure from **complex document formats** (PDFs, Word docs, HTML). For plain text like the State of the Union speech, you can create your own graph relationships.

## Comparison

### PDF with Unstructured (Document 2)

```python
# Unstructured extracts document structure
elements = partition(filename=pdf_file)

# Creates hierarchical relationships
- element_id (Title, Header, Paragraph, etc.)
- parent_id (which section contains this element)
- orig_elements (nested composite elements)

# Graph edges represent document structure
Link.bidir(kind="sections", tag=element_id)
Link.bidir(kind="sections", tag=parent_id)
```

**Graph represents**: Document hierarchy (titles → sections → paragraphs → sentences)

### Plain Text (Your State of Union - Document 1)

```python
# You define relationships based on text characteristics
chunks = text_splitter.split_text(speech_text)

# Create custom relationships
for i, chunk in enumerate(chunks):
    # 1. Sequential (chunks follow each other)
    Link.bidir(kind="sequential", tag=f"chunk_{i-1}")
    Link.bidir(kind="sequential", tag=f"chunk_{i+1}")
    
    # 2. Topic-based (chunks about same subject)
    if "ukraine" in chunk:
        Link.bidir(kind="topic", tag="foreign_policy")
    
    # 3. Self-reference
    Link.bidir(kind="identity", tag=f"chunk_{i}")
```

**Graph represents**: Your custom relationships (sequence, topics, themes)

## Types of Graph Links You Can Create

### 1. Sequential Links (Narrative Flow)
```python
# Chunk N-1 ← → Chunk N ← → Chunk N+1
if i > 0:
    links.append(Link.bidir(kind="sequential", tag=f"chunk_{i-1}"))
if i < len(chunks) - 1:
    links.append(Link.bidir(kind="sequential", tag=f"chunk_{i+1}"))
```
**Use case**: When context flows linearly (speeches, stories, reports)

### 2. Topic-Based Links
```python
# All chunks about "Ukraine" linked together
if "ukraine" in chunk.lower():
    links.append(Link.bidir(kind="topic", tag="ukraine_discussion"))
```
**Use case**: Grouping related content across document

### 3. Entity-Based Links
```python
# All chunks mentioning "Putin"
if "putin" in chunk.lower():
    links.append(Link.bidir(kind="entity", tag="putin"))
```
**Use case**: Tracking references to people, places, organizations

### 4. Semantic Section Links
```python
# Chunks in same paragraph/section
if chunk_starts_new_paragraph:
    current_section_id = f"section_{section_counter}"
links.append(Link.bidir(kind="section", tag=current_section_id))
```
**Use case**: Maintaining original document structure

## How Graph Traversal Works

### Standard Vector Search
```
Query: "What did Biden say about Ukraine?"
  ↓
[Vector similarity search]
  ↓
Top 3 most similar chunks
```

### Graph-Traversal Search
```
Query: "What did Biden say about Ukraine?"
  ↓
[Vector similarity search]
  ↓
Initial top 3 chunks
  ↓
[Follow graph edges with depth=2]
  ↓
• Previous/next chunks (sequential links)
• Other Ukraine chunks (topic links)
• Related entities (entity links)
  ↓
More comprehensive context (5-10+ chunks)
```

## Benefits of Graph Metadata

1. **More Context**: Retrieves related chunks beyond just similarity
2. **Narrative Flow**: Can include preceding/following content
3. **Topic Clustering**: Finds all content on same subject
4. **Better Answers**: LLM gets fuller picture

## Example Comparison

**Question**: "What did Biden say about Ukraine?"

**Standard retrieval (k=3)**:
- Chunk 15: "Putin invaded Ukraine..."
- Chunk 18: "Ukrainian people are brave..."
- Chunk 22: "We support Ukraine..."

**Graph traversal (k=3, depth=2)**:
- Chunk 15: "Putin invaded Ukraine..."
  - → Chunk 14 (sequential: context before)
  - → Chunk 16 (sequential: context after)
  - → Chunk 18 (topic: also about Ukraine)
- Chunk 18: "Ukrainian people are brave..."
  - → Chunk 17 (sequential)
  - → Chunk 22 (topic: Ukraine support)
- Chunk 22: "We support Ukraine..."
  - → Chunk 21, 23 (sequential)

Result: 8-9 chunks with better context flow

## When to Use Each Approach

| Scenario | Approach | Why |
|----------|----------|-----|
| PDFs with structure | Unstructured + Graph | Preserves document hierarchy |
| Plain text/transcripts | Custom Graph Links | Define your own relationships |
| Simple Q&A | Standard Vector Search | Faster, sufficient for basic needs |
| Complex analysis | Graph Traversal | More context, better understanding |
| Short documents | Standard Vector Search | Graph overhead not worth it |
| Large documents | Graph Traversal | Navigate complex information |

## Implementation in Your Code

The new script (`rag_demo_graph.py`) adds:

1. **Graph link creation** during document ingestion
2. **Two retrieval methods** for comparison:
   - Standard: `search_kwargs={"k": 3}`
   - Graph: `search_type="traversal", search_kwargs={"k": 3, "depth": 2}`
3. **Side-by-side comparison** of results

## Next Steps

Enhance with:
- **Named entity recognition** (spaCy, NLTK) for automatic entity linking
- **Topic modeling** (LDA, BERTopic) for automatic topic detection  
- **Sentence boundaries** for paragraph-level linking
- **Semantic similarity** between chunks for content-based linking

## Summary

**No need to use unstructured for plain text.** You can create meaningful graph relationships based on:
- Sequence (what comes before/after)
- Topics (what's discussed where)
- Entities (who/what is mentioned)
- Custom logic (your domain knowledge)

The graph metadata enhances retrieval by helping the system understand relationships between chunks beyond just vector similarity.

# RAG and Graph RAG Explained 

## Phase 1: Original Text → Chunks → Embeddings → Vector Database

**Text Chunking**
```
"Madam Speaker, Madam Vice President... [5000 words]"
↓
Chunk 1: "Madam Speaker, Madam Vice President..." (500 words)
Chunk 2: "Six days ago, Russia's Vladimir Putin..." (500 words)
Chunk 3: "Tonight, I'm announcing a crackdown..." (500 words)
```

**Convert to Embeddings (Vectors)**
```
Chunk 1 text → Embedding Model → [0.23, -0.45, 0.89, ..., 0.12]
                                  (384 numbers for MiniLM)
```

Embeddings are a mathematical representation** of the text's meaning. Text is given numeric dimensions (vectors). Similar meanings have numbers 'close' to each other.
```
Example (simplified to 2D):
"Ukraine war"     → [0.8, 0.9]  ← These two phrases are close together
"Russia conflict" → [0.82, 0.88] ← in vector space

"Ice cream"       → [-0.5, 0.2]  ← but far away from the meaning of this text
```

## Phase 2: Query Time (Every Question)

**Step 1: Convert Question to Embedding**
```
User asks: "What did Biden say about Ukraine?"
↓
Embedding Model
↓
Query Vector: [0.81, 0.89, -0.23, ..., 0.45]
```

**Step 2: Similarity Search**
```
Compare query vector to ALL chunk vectors using cosine similarity:

Query:   [0.81, 0.89, ...]
Chunk 1: [0.23, -0.45, ...] → Similarity: 0.34 (not relevant)
Chunk 2: [0.79, 0.91, ...]  → Similarity: 0.92 (very relevant!)
Chunk 3: [0.15, -0.12, ...] → Similarity: 0.18 (not relevant)

Pick top 3 most similar chunks (top_k setting)

The text of the returned chunks are joined together to create a piece of context text
```

**Step 3: Add the context to question sent to LLM**
```
The LLM receives the context text, not embeddings (numbers)

Prompt to Claude:
┌─────────────────────────────────────────────┐
│ Context (retrieved text):                   │
│ "Six days ago, Russia's Vladimir Putin      │
│ sought to shake the foundations of the free │
│ world... He met the Ukrainian people..."    │
│                                             │
│ Question: What did Biden say about Ukraine? │
│                                             │
│ Answer based on the context above:          │
└─────────────────────────────────────────────┘
```

**Step 4: LLM Response**
```
LLM reads the text and generates an answer
↓
"Biden praised the Ukrainian people's courage
in resisting Putin's invasion..."
```

## Key Insights

### Embeddings are for Finding, not for Reading
```
┌─────────────────────────────────────────────────┐
│ Embeddings (vectors): Used ONLY for search      │
│ - Query vector → Find similar chunk vectors     │
│ - i.e. "Which chunks are relevant?"             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Original Text: Sent to the LLM                  │
│ - Actual words from matching chunks             │
│ - LLM reads and understands like humans do      │
└─────────────────────────────────────────────────┘
```

### Two Separate AI Models at Work

1. **Embedding Model** (e.g., MiniLM)
   - Small, specialized model
   - Job: Convert text ↔ vectors
   - Used for: Similarity search only
   
2. **Large Language Model** (e.g., Claude)
   - Large, general model
   - Job: Understand text and generate responses
   - Used for: Reading context and answering questions

## Visual Example
```
USER QUESTION: "What about inflation?"

Step 1: Embedding Model
"What about inflation?" → [0.45, -0.23, 0.89, ...]

Step 2: Vector Search
Database has 100 chunks with vectors
↓
Find 3 most similar vectors
↓
Get their IDs: [chunk_42, chunk_17, chunk_88]

Step 3: Retrieve Original Text
chunk_42: "Tonight, I'm announcing a crackdown on..."
chunk_17: "We have a choice. One way to fight inflation..."  
chunk_88: "Lower your costs, not your wages..."

Step 4: Build Prompt
Context: [actual text from chunks above]
Question: "What about inflation?"

Step 5: Send to LLM
LLM reads the text and responds

Step 6: LLM's Response
"Biden announced several measures to fight inflation,
including a crackdown on shipping costs..."
```

## RAG compared to vanilla LLM

**Vanilla LLM:**
```
Question: "What did Biden say about inflation?"
↓
LLM's training data (might not include this specific speech)
↓
Generic answer or "I don't have that information"
```

**RAG-Enhanced:**
```
Question: "What did Biden say about inflation?"
↓
Semantic search finds EXACT relevant paragraphs
↓
LLM reads those specific paragraphs
↓
Accurate, specific answer with exact details
```

## The Math Behind Similarity

Cosine similarity measures the angle between vectors:
```
Vector A: [1, 0]  ─────→
                    \ 45°
Vector B: [0.7, 0.7] ─→

Cosine similarity = cos(45°) = 0.707

Similar meaning = Small angle = High score (close to 1)
Different meaning = Large angle = Low score (close to 0)
```

## How Graph Traversal Works (Graph RAG)

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


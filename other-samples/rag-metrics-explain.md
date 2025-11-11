# RAG Metrics Explanation

## Quantitative Metrics to Prove Graph RAG is Better

## 1. Chunks Retrieved

**What it measures:** Raw number of document chunks retrieved

```
Standard: 3 chunks
Enhanced: 7 chunks
Increase: +133%
```

**Why it matters:** More chunks = more context for the LLM to work with

**Good result:** 50%+ increase shows graph traversal is finding additional relevant content

## 2. Context Size (characters)

**What it measures:** Total characters of text sent to the LLM

```
Standard: 2,500 characters
Enhanced: 5,800 characters  
Increase: +132%
```

**Why it matters:** 
- More context = more complete picture
- LLM can synthesize from broader information
- Reduces hallucination by providing more facts

**Good result:** 100%+ increase means significantly richer context

## 3. Unique Topics Covered

**What it measures:** Number of distinct topics in the retrieved chunks

```
Standard: 2 topics (AI, healthcare)
Enhanced: 4 topics (AI, healthcare, robotics, education)
Increase: +2 topics
```

**Why it matters:**
- Shows breadth of information
- Cross-topic questions get better coverage
- More comprehensive answers

**Good result:** +1 or more topics shows better coverage

## 4. Chunk Spread (contextual diversity)

**What it measures:** Range of chunk IDs retrieved (max - min)

```
Standard: Chunks [5, 6, 7] → Spread = 2
Enhanced: Chunks [3, 4, 5, 6, 7, 8, 12] → Spread = 9
Increase: +7 chunks
```

**Why it matters:**
- Higher spread = pulling from different document sections
- Low spread = all context from same area (might miss relevant info elsewhere)
- Graph traversal finds related content across the document

**Good result:** +3 or more shows graph is finding diverse sources

## 5. Answer Detail (answer length)

**What it measures:** Character length of LLM's response

```
Standard: 450 characters
Enhanced: 720 characters
Increase: +60%
```

**Why it matters:**
- More context → more detailed answers
- LLM can provide specific examples
- Better explanations with more supporting info

**Good result:** 30%+ increase shows LLM is leveraging the extra context

## Example Output:

```
================================================================================
METRICS COMPARISON:
────────────────────────────────────────────────────────────────────────────────
Chunks Retrieved:         3 → 7 (+133.3%)
Context Size (chars):     2,458 → 5,823 (+137.0%)
Unique Topics Covered:    2 → 4 (+2)
Chunk Spread (diversity): 2 → 9 (+7 chunks)
Answer Length (chars):    451 → 715 (+58.5%)
```

## Aggregate Summary:

After processing all questions, you'll see:

```
================================================================================
AGGREGATE METRICS ACROSS ALL QUESTIONS
================================================================================

Average Chunk Retrieval Increase:    +120.5%
Average Context Size Increase:       +115.3%
Average Topic Coverage Increase:     +1.7 topics
Average Chunk Spread Increase:       +5.3 chunks
Average Answer Detail Increase:      +45.2%

────────────────────────────────────────────────────────────────────────────────
INTERPRETATION:
────────────────────────────────────────────────────────────────────────────────
✅ Metadata-enhanced retrieval provides SIGNIFICANTLY more context
✅ Metadata-enhanced covers 1.7 more topics on average
✅ Metadata-enhanced retrieves from 5.3 more diverse locations
✅ Metadata-enhanced produces 45.2% more detailed answers
```

## What Good Results Look Like:

### Excellent Results (Clear Win for Graph RAG):
- Chunk increase: **100%+**
- Context increase: **100%+**
- Topic increase: **+2 or more**
- Spread increase: **+5 or more**
- Answer increase: **50%+**

### Good Results (Graph RAG Helps):
- Chunk increase: **50-100%**
- Context increase: **50-100%**
- Topic increase: **+1**
- Spread increase: **+3-5**
- Answer increase: **30-50%**

### Marginal Results (Document Too Small):
- Chunk increase: **<50%**
- Context increase: **<50%**
- Topic increase: **0**
- Spread increase: **<3**
- Answer increase: **<30%**

## Statistical Significance:

For a truly rigorous comparison, you could also add:

### 1. **Precision/Recall with Ground Truth**
```python
# If you have labeled relevant chunks for each question
relevant_chunks = {
    "How is AI used?": [2, 3, 15, 18, 22, 45, 67, 88],
}

precision = len(retrieved_relevant) / len(retrieved_total)
recall = len(retrieved_relevant) / len(all_relevant)
f1_score = 2 * (precision * recall) / (precision + recall)
```

### 2. **Answer Quality Score (Human Evaluation)**
```python
# Rate answers 1-5 on:
- Completeness
- Accuracy  
- Specificity
- Coherence
```

### 3. **Semantic Similarity to Ground Truth**
```python
from sentence_transformers import SentenceTransformer, util

# Compare answer to reference answer
similarity = util.cos_sim(answer_embedding, reference_embedding)
```

### 4. **Token Efficiency**
```python
# Measure information density
info_density = answer_quality_score / context_tokens_used
# Higher = more efficient use of context
```

## Why These Metrics Matter:

1. **Chunks Retrieved** - Direct measure of graph traversal effectiveness
2. **Context Size** - Shows if you're actually getting more information
3. **Topic Coverage** - Proves cross-topic/cross-section retrieval works
4. **Chunk Spread** - Demonstrates diversity (not just grabbing adjacent chunks)
5. **Answer Detail** - End-to-end validation that more context → better answers

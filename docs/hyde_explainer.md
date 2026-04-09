# HyDE vs Standard Retrieval

## Standard Bi-Encoder Retrieval

```
User Query ──► [Bi-Encoder] ──► query vector
                                     │
                                     ▼ cosine similarity
Document DB ──► [Bi-Encoder] ──► doc vectors ──► top-k results
```

**Problem**: query vector ≠ document vector region in embedding space, even for semantically equivalent content.

## HyDE Retrieval

```
User Query ──► [LLM: generate fake answer] ──► Hypothetical Doc
                                                      │
                                                      ▼
                                              [Bi-Encoder]
                                                      │
                                                      ▼ cosine similarity
Document DB ──► [Bi-Encoder] ──────────────► doc vectors ──► top-k results
```

**Solution**: the LLM acts as a vocabulary bridge. Its output uses the same academic/technical register as the documents in the database.

## Example

| | Text |
|---|---|
| Raw query | "how does attention work?" |
| HyDE output | "The attention mechanism operates by computing a weighted sum of value vectors, where weights are determined by the compatibility between query and key representations via scaled dot-product scoring..." |
| Target passage | "We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values..." |

The HyDE output and the target passage are now vocabulary-aligned, leading to much higher cosine similarity in the embedding space.

## When HyDE helps most

- Domain-specific corpora (legal, medical, academic)
- Short queries against long documents
- Cases where users don't know the "right" terminology

## When HyDE may not help

- Factual lookup queries where the exact entity name is known ("what year was the transformer paper published?")
- Very small corpora where any chunk will be retrieved regardless
- Latency-constrained applications (adds one LLM call per query)

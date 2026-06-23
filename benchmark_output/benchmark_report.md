# Flat RAG vs GraphRAG benchmark

Corpus: 70 documents. Questions: 20. Same LLM for answer generation.

| Metric | Flat RAG | GraphRAG |
|---|---:|---:|
| Mean expected-document recall | 32.8% | 34.2% |
| Mean retrieval latency | 4.98 ms | 8.42 ms |
| Mean generation latency | 3838.51 ms | 1917.42 ms |
| Mean context size | 7232 chars | 2769 chars |

Recall wins: Flat 8, GraphRAG 7, ties 5.

## Graph construction cost

- Existing graph input: 2,630 triples from `dataset/triples/graphrag_edges_neo4j.csv`
  (371,661 bytes).
- Loading triples into NetworkX took 25.51 ms and wrote a
  741,540-byte GraphML file. No database service is required.
- Building from raw text would require 3,421 extraction calls,
  covering about 4,070,797 prompt characters before response tokens.
- GraphRAG adds extraction, normalization, persistence, and refresh cost. Flat RAG only
  chunks and indexes text, so it is materially cheaper to build and update.
- The graph cost is justified when questions require joining relationships across
  documents; for direct factual lookup, Flat RAG is usually the cheaper baseline.

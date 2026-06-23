# Flat RAG vs GraphRAG benchmark

Corpus: 70 documents. Questions: 20. Same LLM for answer generation.

| Metric | Flat RAG | GraphRAG |
|---|---:|---:|
| Mean expected-document recall | 32.8% | 28.1% |
| Mean retrieval latency | 5.08 ms | 18.03 ms |
| Mean generation latency | 3920.75 ms | 2462.41 ms |
| Mean context size | 7232 chars | 10526 chars |

Recall wins: Flat 7, GraphRAG 4, ties 9.

## Graph construction cost

- Existing graph input: 3,394 triples from `dataset/triples/graphrag_edges_neo4j.csv`
  (1,725,191 bytes).
- Loading triples into NetworkX took 78.83 ms and wrote a
  2,091,889-byte GraphML file. No database service is required.
- Building from raw text would require 3,421 extraction calls,
  covering about 4,070,797 prompt characters before response tokens.
- GraphRAG adds extraction, normalization, persistence, and refresh cost. Flat RAG only
  chunks and indexes text, so it is materially cheaper to build and update.
- The graph cost is justified when questions require joining relationships across
  documents; for direct factual lookup, Flat RAG is usually the cheaper baseline.

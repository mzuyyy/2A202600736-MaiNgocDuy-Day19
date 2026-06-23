# Local GraphRAG with NetworkX

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
python graphrag_local.py check
python graphrag_local.py ingest --max-docs 1
python graphrag_local.py ask "What drives electric vehicle market growth?"
python benchmark_rag.py
python visualize_graph.py --output benchmark_output/graph.png --max-nodes 50
```

The program reads text files from `dataset/`, extracts triples with
`langextract`, writes them to `dataset/triples/graphrag_edges_langextract.csv`,
stores the graph in `graph.graphml`, and reads these values from `.env`:

```dotenv
GRAPH_PATH=graph.graphml
TRIPLES_PATH=dataset/triples/graphrag_edges_langextract.csv
```

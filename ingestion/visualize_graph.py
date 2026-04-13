"""Generate LangGraph visualization for the ingestion subgraph."""

import importlib.util
import json
import pathlib

from ingestion.graph import build_ingestion_graph


def generate_graph_visualization():
    """Generate and save the ingestion graph visualization as PNG."""
    graph = build_ingestion_graph().compile()

    try:
        langgraph_graph = graph.get_graph()
        docs_dir = pathlib.Path(__file__).parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)

        # Try mermaid first if IPython is available
        if importlib.util.find_spec("IPython"):
            mermaid_code = langgraph_graph.draw_mermaid()
            mermaid_path = docs_dir / "ingestion_graph.mmd"
            mermaid_path.write_text(mermaid_code)
        else:
            graph_data = langgraph_graph.to_json()
            graph_path = docs_dir / "ingestion_graph.json"
            graph_path.write_text(json.dumps(graph_data, indent=2))

    except Exception:
        pass

    return graph


if __name__ == "__main__":
    generate_graph_visualization()

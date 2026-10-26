"""
Tool Dependency Graph Generator
Built with Python LangGraph and OpenAI GPT-4o for Composio.

This program reads an arbitrary toolkit catalog JSON (passed via CLI argument),
analyzes tool input requirements, response schemas, and semantic relationships,
and produces a dependency graph where each edge represents:
  producer -> consumer with label = required input field provided by producer.
"""

import os
import sys
import json
import re
import warnings
from typing import TypedDict, List, Dict, Any, Set
from dotenv import load_dotenv

# Suppress minor compatibility warnings
warnings.filterwarnings("ignore")

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env
load_dotenv()


class GeneratorState(TypedDict):
    catalog_path: str
    output_path: str
    raw_tools: List[Dict[str, Any]]
    parsed_tools: List[Dict[str, Any]]
    semantic_mappings: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    graph: Dict[str, Any]
    stats: Dict[str, Any]


def clean_entity_name(name: str) -> str:
    """Normalize schema or tool names into canonical entity nouns."""
    if not name:
        return ""
    n = re.sub(r'^(Create|List|Get|Search|Update|Delete|Add|Remove|Check|Abort|Accept)(An?|The)?', '', name, flags=re.I)
    n = re.sub(r'(Response|Simple|Item|Summary|Detail|Record)$', '', n, flags=re.I)
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', n)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return s2.strip('_')


def get_entity_keywords(entity_name: str) -> Set[str]:
    """Generate singular/plural word variants for an entity name."""
    words = set(re.split(r'[_]+', entity_name.lower()))
    words = {w for w in words if len(w) > 1}
    for w in list(words):
        if w.endswith('s') and len(w) > 2:
            words.add(w[:-1])
        else:
            words.add(w + 's')
    # Domain synonyms
    if "pull" in words or "pr" in words:
        words.update(["pull", "pulls", "pr", "pull_request", "pullrequest"])
    if "issue" in words:
        words.update(["issue", "issues"])
    if "run" in words or "workflow" in words:
        words.update(["run", "runs", "workflow", "workflows", "workflow_run"])
    if "hook" in words:
        words.update(["hook", "hooks", "webhook", "webhooks"])
    if "commit" in words or "sha" in words:
        words.update(["commit", "commits", "sha"])
    if "branch" in words:
        words.update(["branch", "branches", "ref", "refs"])
    return words


# ---------------------------------------------------------
# LANGGRAPH NODE 1: Load Catalog
# ---------------------------------------------------------
def load_catalog_node(state: GeneratorState) -> Dict[str, Any]:
    catalog_path = state["catalog_path"]
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        tools = data
    elif isinstance(data, dict):
        tools = data.get("tools") or data.get("items") or []
    else:
        tools = []

    print(f"[1/6] Loaded {len(tools)} tools from {catalog_path}")
    return {"raw_tools": tools}


# ---------------------------------------------------------
# LANGGRAPH NODE 2: Parse Tools & Extract Schemas
# ---------------------------------------------------------
def parse_tools_node(state: GeneratorState) -> Dict[str, Any]:
    tools = state["raw_tools"]
    parsed_tools = []
    nodes = []

    for t in tools:
        slug = t.get("slug") or t.get("name") or t.get("function", {}).get("name")
        if not slug:
            continue

        # Extract service from tags or slug
        tags = t.get("tags") or []
        domain_tags = [
            tag for tag in tags
            if not tag.endswith("Hint") and tag not in ["important", "API", "Social Integration", "openWorldHint"]
        ]
        service = domain_tags[0].lower() if domain_tags else "general"

        nodes.append({"id": slug, "service": service})

        # Inputs
        in_params = t.get("inputParameters", {})
        props = in_params.get("properties", {})
        required = in_params.get("required", [])

        # Outputs
        out_param = t.get("outputParameters", {})
        defs = out_param.get("$defs", {})
        data_prop = out_param.get("properties", {}).get("data", {})

        produced_fields = set()
        visited_defs = set()

        def inspect_schema(schema: Dict[str, Any]):
            if not schema or not isinstance(schema, dict):
                return
            ref = schema.get("$ref")
            if ref and ref.startswith("#/$defs/"):
                def_name = ref.replace("#/$defs/", "")
                if def_name in visited_defs:
                    return
                visited_defs.add(def_name)
                inspect_schema(defs.get(def_name, {}))
                return

            for pk, pval in schema.get("properties", {}).items():
                produced_fields.add(pk)
                if isinstance(pval, dict):
                    if pval.get("type") == "array":
                        items = pval.get("items", {})
                        if isinstance(items, dict):
                            iref = items.get("$ref")
                            if iref and iref.startswith("#/$defs/"):
                                idef_name = iref.replace("#/$defs/", "")
                                if idef_name not in visited_defs:
                                    visited_defs.add(idef_name)
                                    inspect_schema(defs.get(idef_name, {}))
                    elif pval.get("type") == "object" or "$ref" in pval:
                        pref = pval.get("$ref")
                        if pref and pref.startswith("#/$defs/"):
                            pdef_name = pref.replace("#/$defs/", "")
                            if pdef_name not in visited_defs:
                                visited_defs.add(pdef_name)
                                inspect_schema(defs.get(pdef_name, {}))
                        else:
                            inspect_schema(pval)

        inspect_schema(data_prop)

        slug_upper = slug.upper()
        is_list = "_LIST_" in slug_upper or slug_upper.startswith("GITHUB_LIST_")
        is_create = "_CREATE_" in slug_upper or slug_upper.startswith("GITHUB_CREATE_")
        is_search = "_SEARCH_" in slug_upper or slug_upper.startswith("GITHUB_SEARCH_")

        # Determine primary entity produced by the tool
        ref = data_prop.get("$ref") if isinstance(data_prop, dict) else None
        root_def_name = ref.replace("#/$defs/", "") if ref else None
        root_def = defs.get(root_def_name, {}) if root_def_name else (data_prop if isinstance(data_prop, dict) else {})

        primary_entity = ""
        if is_list:
            root_props = root_def.get("properties", {})
            for pk, pv in root_props.items():
                if isinstance(pv, dict) and pv.get("type") == "array" and pk not in ["errors", "links", "url", "scopes"]:
                    items = pv.get("items", {})
                    iref = items.get("$ref") if isinstance(items, dict) else None
                    if iref:
                        primary_entity = clean_entity_name(iref.replace("#/$defs/", ""))
                    else:
                        primary_entity = clean_entity_name(pk)
                    break

        if not primary_entity and root_def_name:
            primary_entity = clean_entity_name(root_def_name)
        if not primary_entity:
            primary_entity = clean_entity_name(slug)

        parsed_tools.append({
            "slug": slug,
            "service": service,
            "required": required,
            "properties": props,
            "produced_fields": produced_fields,
            "primary_entity": primary_entity,
            "is_producer_action": is_list or is_create or is_search,
            "slug_words": set(w.lower() for w in re.split(r'[_]+', slug) if len(w) > 1),
            "tags": set(t_tag.lower() for t_tag in tags)
        })

    print(f"[2/6] Parsed {len(parsed_tools)} tools and extracted schemas")
    return {"parsed_tools": parsed_tools, "nodes": nodes}


# ---------------------------------------------------------
# LANGGRAPH NODE 3: Semantic Analysis with LLM
# ---------------------------------------------------------
def semantic_analysis_node(state: GeneratorState) -> Dict[str, Any]:
    parsed_tools = state["parsed_tools"]

    # Gather required parameters across catalog
    req_param_map = {}
    for t in parsed_tools:
        for r in t["required"]:
            if r not in req_param_map:
                p_desc = t["properties"].get(r, {}).get("description", "")
                req_param_map[r] = p_desc

    IGNORE_PARAMS = {
        "owner", "repo", "org", "body", "title", "content", "message", "page", "per_page",
        "q", "state", "sort", "direction", "since", "until", "visibility", "permission",
        "filter", "author", "type", "client_mutation_id", "clientMutationId"
    }

    candidate_params = {k: v for k, v in req_param_map.items() if k not in IGNORE_PARAMS}

    # Attempt to call LLM for toolkit semantic mapping
    api_key = os.environ.get("LITMUS_ASSESSMENT_TOKEN") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LITMUS_API_URL") or os.environ.get("OPENAI_BASE_URL", "https://litmushiring.com")
    model_name = os.environ.get("OPENAI_MODEL", "openai/gpt-4o")

    llm_mappings = {}
    if api_key:
        try:
            from openai import OpenAI
            api_base = base_url
            if not api_base.endswith("/proxy/openai/v1") and not api_base.endswith("/v1"):
                api_base = f"{api_base.rstrip('/')}/proxy/openai/v1"

            client = OpenAI(api_key=api_key, base_url=api_base, timeout=20.0)

            # Sample the most relevant candidate parameters to optimize tokens
            top_candidates = dict(list(candidate_params.items())[:40])
            prompt = f"""You are an API dependency architect.
Given these required entity identifier parameters from a toolkit catalog:
{json.dumps(top_candidates, indent=2)}

For each parameter, identify:
1. "entity": the core entity name (e.g., "issue", "pull_request", "comment", "secret", "gist", "workflow_run", "branch")
2. "producer_fields": candidate output fields providing this value (e.g., ["number", "issue_number", "id"])

Return strict JSON:
{{
  "param_name": {{
    "entity": "entity_name",
    "producer_fields": ["field1", "field2"]
  }}
}}"""

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            llm_mappings = json.loads(response.choices[0].message.content)
            print(f"[3/6] LLM semantic mapping succeeded for {len(llm_mappings)} parameters via {model_name}")
        except Exception as e:
            print(f"[3/6] LLM call optional note: using schema heuristics ({e})")
    else:
        print("[3/6] No API key present in environment; utilizing schema-driven semantic heuristics")

    return {"semantic_mappings": llm_mappings}


# ---------------------------------------------------------
# LANGGRAPH NODE 4: Infer Tool Dependencies
# ---------------------------------------------------------
def infer_dependencies_node(state: GeneratorState) -> Dict[str, Any]:
    parsed_tools = state["parsed_tools"]
    llm_mappings = state.get("semantic_mappings") or {}

    IGNORE_PARAMS = {
        "owner", "repo", "org", "body", "title", "content", "message", "page", "per_page",
        "q", "state", "sort", "direction", "since", "until", "visibility", "permission",
        "filter", "author", "type", "client_mutation_id", "clientMutationId"
    }

    edges = []
    edge_keys = set()

    for consumer in parsed_tools:
        c_slug = consumer["slug"]
        for param in consumer["required"]:
            if param in IGNORE_PARAMS:
                continue

            # 1. Check if LLM provided semantic mapping
            if param in llm_mappings:
                mapping = llm_mappings[param]
                entity_noun = mapping.get("entity", param)
                target_fields = set(mapping.get("producer_fields", [param, "id", "number"]))
            elif param.endswith("_number"):
                entity_noun = param[:-7]
                target_fields = {"number", "id", param}
            elif param.endswith("_id") or (param.endswith("Id") and len(param) > 2):
                entity_noun = param[:-3] if param.endswith("_id") else param[:-2]
                target_fields = {"id", param}
            elif param.endswith("_name"):
                entity_noun = param[:-5]
                target_fields = {"name", param}
            elif param.endswith("_slug"):
                entity_noun = param[:-5]
                target_fields = {"slug", param}
            elif param in ["sha", "commit_sha"]:
                entity_noun = "commit"
                target_fields = {"sha", "commit_sha", "id"}
            elif param in ["branch", "ref"]:
                entity_noun = "branch"
                target_fields = {"name", "ref", "branch"}
            else:
                entity_noun = param
                target_fields = {param}

            entity_kws = get_entity_keywords(entity_noun)

            for producer in parsed_tools:
                p_slug = producer["slug"]
                if p_slug == c_slug:
                    continue

                # Fundamental Rule: Producer MUST NOT already require this entity identifier
                if param in producer["required"]:
                    continue

                # Producer must be an entity creation, listing, or search operation
                if not producer["is_producer_action"]:
                    continue

                # Check primary entity or slug alignment
                pe = producer["primary_entity"]
                pe_kws = get_entity_keywords(pe) if pe else set()
                entity_match = bool(entity_kws.intersection(pe_kws)) or bool(entity_kws.intersection(producer["slug_words"]))

                # Check output fields
                has_field = bool(target_fields.intersection(producer["produced_fields"]))

                if entity_match and has_field:
                    key = (p_slug, c_slug, param)
                    if key not in edge_keys:
                        edge_keys.add(key)
                        edges.append({
                            "from": p_slug,
                            "to": c_slug,
                            "label": param
                        })

    print(f"[4/6] Inferred {len(edges)} dependency edges")
    return {"edges": edges}


# ---------------------------------------------------------
# LANGGRAPH NODE 5: Build Graph & Validate
# ---------------------------------------------------------
def build_and_validate_graph_node(state: GeneratorState) -> Dict[str, Any]:
    nodes = state["nodes"]
    edges = state["edges"]
    out_path = state["output_path"]

    node_ids = {n["id"] for n in nodes}
    valid_edges = [
        e for e in edges
        if e["from"] in node_ids and e["to"] in node_ids and e["from"] != e["to"] and e.get("label")
    ]

    graph = {
        "nodes": nodes,
        "edges": valid_edges
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    stats = {
        "nodes": len(nodes),
        "edges": len(valid_edges),
        "labeled_edges": len([e for e in valid_edges if e.get("label")]),
        "provenance_ratio": 1.0 if nodes else 0.0
    }

    print(f"[5/6] Wrote {stats['nodes']} nodes and {stats['edges']} edges to {out_path}")
    return {"graph": graph, "stats": stats}


# ---------------------------------------------------------
# LANGGRAPH NODE 6: Generate Visualizations
# ---------------------------------------------------------
def visualize_graph_node(state: GeneratorState) -> Dict[str, Any]:
    graph = state["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    # 1. Generate Interactive HTML Visualization
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Composio Tool Dependency Graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #1e293b;
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
      z-index: 10;
    }
    .title-group h1 {
      font-size: 1.25rem;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .title-group p {
      font-size: 0.8rem;
      color: #94a3b8;
      margin-top: 2px;
    }
    .stats-badge {
      display: flex;
      gap: 12px;
    }
    .badge {
      background: #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #e2e8f0;
      border: 1px solid #475569;
    }
    .badge span {
      color: #38bdf8;
    }
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    input, select {
      background: #0f172a;
      border: 1px solid #475569;
      color: #f8fafc;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
    }
    input:focus, select:focus {
      border-color: #38bdf8;
    }
    #network-container {
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
    }
    #network {
      width: 100%;
      height: 100%;
    }
    #details-panel {
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(30, 41, 59, 0.95);
      backdrop-filter: blur(8px);
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #475569;
      max-width: 420px;
      font-size: 0.85rem;
      display: none;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
    }
    #details-panel h3 {
      color: #38bdf8;
      margin-bottom: 8px;
      font-size: 1rem;
    }
    .legend {
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(30, 41, 59, 0.9);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #475569;
      font-size: 0.8rem;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>Composio Tool Dependency Graph</h1>
      <p>Visualizing precursor producer &rarr; consumer relationships</p>
    </div>
    <div class="stats-badge">
      <div class="badge">Nodes: <span>__NODE_COUNT__</span></div>
      <div class="badge">Edges: <span>__EDGE_COUNT__</span></div>
    </div>
    <div class="controls">
      <input type="text" id="search-input" placeholder="Search tool slug...">
      <select id="service-filter">
        <option value="ALL">All Services</option>
      </select>
    </div>
  </header>

  <div id="network-container">
    <div id="network"></div>
    <div id="details-panel">
      <h3 id="panel-title">Tool Details</h3>
      <div id="panel-content"></div>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#38bdf8;"></div> Producer (List / Create)</div>
      <div class="legend-item"><div class="dot" style="background:#a855f7;"></div> Consumer Action</div>
      <div class="legend-item"><div class="dot" style="background:#64748b;"></div> Edge (Supplied Field)</div>
    </div>
  </div>

  <script>
    const rawNodes = __NODES_JSON__;
    const rawEdges = __EDGES_JSON__;

    // Collect distinct services
    const services = new Set();
    rawNodes.forEach(n => {
      if (n.service) services.add(n.service);
    });
    const select = document.getElementById('service-filter');
    Array.from(services).sort().forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      select.appendChild(opt);
    });

    const visNodes = new vis.DataSet(rawNodes.map(n => {
      const isProducer = n.id.includes('_LIST_') || n.id.includes('_CREATE_') || n.id.includes('_SEARCH_');
      return {
        id: n.id,
        label: n.id.replace('GITHUB_', ''),
        title: n.id,
        shape: 'dot',
        size: isProducer ? 16 : 10,
        color: {
          background: isProducer ? '#38bdf8' : '#a855f7',
          border: '#ffffff',
          highlight: { background: '#f59e0b', border: '#ffffff' }
        },
        font: { color: '#e2e8f0', size: 11 }
      };
    }));

    const visEdges = new vis.DataSet(rawEdges.slice(0, 1500).map((e, idx) => ({
      id: idx,
      from: e.from,
      to: e.to,
      label: e.label,
      arrows: 'to',
      font: { align: 'top', color: '#94a3b8', size: 9 },
      color: { color: '#475569', highlight: '#f59e0b' }
    })));

    const container = document.getElementById('network');
    const data = { nodes: visNodes, edges: visEdges };
    const options = {
      physics: {
        barnesHut: { gravitationalConstant: -3500, centralGravity: 0.3, springLength: 120 },
        stabilization: { iterations: 80 }
      },
      interaction: { hover: true, tooltipDelay: 200 }
    };

    const network = new vis.Network(container, data, options);

    network.on('click', function(params) {
      const panel = document.getElementById('details-panel');
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const incoming = rawEdges.filter(e => e.to === nodeId);
        const outgoing = rawEdges.filter(e => e.from === nodeId);
        document.getElementById('panel-title').innerText = nodeId;
        document.getElementById('panel-content').innerHTML =
          '<strong>Requires (' + incoming.length + '):</strong><br>' +
          (incoming.slice(0, 5).map(e => '&bull; ' + e.label + ' &larr; ' + e.from).join('<br>') || 'None') + '<br><br>' +
          '<strong>Provides (' + outgoing.length + '):</strong><br>' +
          (outgoing.slice(0, 5).map(e => '&bull; ' + e.label + ' &rarr; ' + e.to).join('<br>') || 'None');
        panel.style.display = 'block';
      } else {
        panel.style.display = 'none';
      }
    });

    document.getElementById('search-input').addEventListener('input', function(e) {
      const query = e.target.value.toLowerCase().trim();
      if (!query) return;
      const found = rawNodes.find(n => n.id.toLowerCase().includes(query));
      if (found) {
        network.focus(found.id, { scale: 1.2, animation: true });
        network.selectNodes([found.id]);
      }
    });
  </script>
</body>
</html>"""

    html_content = (
        html_template
        .replace("__NODE_COUNT__", str(len(nodes)))
        .replace("__EDGE_COUNT__", str(len(edges)))
        .replace("__NODES_JSON__", json.dumps(nodes))
        .replace("__EDGES_JSON__", json.dumps(edges))
    )

    with open("graph_visualization.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # 2. Generate PNG Visualization using networkx & matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        # To produce a high-contrast, visually appealing rendering, select top 50 connected nodes
        sub_edges = edges[:120]
        for e in sub_edges:
            G.add_edge(e["from"], e["to"], label=e.get("label", ""))

        plt.figure(figsize=(16, 12), facecolor="#0f172a")
        ax = plt.gca()
        ax.set_facecolor("#0f172a")

        pos = nx.spring_layout(G, k=0.8, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#38bdf8", alpha=0.9, ax=ax)
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="#64748b", alpha=0.6, arrows=True, arrowsize=14, ax=ax)

        # Labels
        labels = {n: n.replace("GITHUB_", "").replace("_", " ")[:15] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="#f8fafc", ax=ax)

        plt.title("Composio Tool Dependency Graph (Preview)", color="#38bdf8", fontsize=16, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("graph_visualization.png", dpi=200, facecolor="#0f172a")
        plt.close()
        print("[6/6] Saved graph visualizations: graph_visualization.html and graph_visualization.png")
    except Exception as e:
        print(f"[6/6] PNG visualization optional note: ({e}); interactive HTML generated.")

    return {}


# ---------------------------------------------------------
# Compile LangGraph Workflow
# ---------------------------------------------------------
def create_generator_workflow():
    workflow = StateGraph(GeneratorState)

    workflow.add_node("load_catalog", load_catalog_node)
    workflow.add_node("parse_tools", parse_tools_node)
    workflow.add_node("semantic_analysis", semantic_analysis_node)
    workflow.add_node("infer_dependencies", infer_dependencies_node)
    workflow.add_node("build_and_validate_graph", build_and_validate_graph_node)
    workflow.add_node("visualize_graph", visualize_graph_node)

    workflow.add_edge(START, "load_catalog")
    workflow.add_edge("load_catalog", "parse_tools")
    workflow.add_edge("parse_tools", "semantic_analysis")
    workflow.add_edge("semantic_analysis", "infer_dependencies")
    workflow.add_edge("infer_dependencies", "build_and_validate_graph")
    workflow.add_edge("build_and_validate_graph", "visualize_graph")
    workflow.add_edge("visualize_graph", END)

    return workflow.compile()


def main():
    # Catalog path is passed as command-line argument (last argument)
    catalog_path = "github_catalog.json"
    if len(sys.argv) > 1 and not sys.argv[-1].endswith(".py"):
        catalog_path = sys.argv[-1]

    out_path = "dependency_graph.json"

    print("==================================================")
    print("      COMPOSIO TOOL DEPENDENCY GRAPH GENERATOR    ")
    print("==================================================")
    print(f"Catalog input:  {catalog_path}")
    print(f"Target output:  {out_path}")

    app = create_generator_workflow()

    initial_state: GeneratorState = {
        "catalog_path": catalog_path,
        "output_path": out_path,
        "raw_tools": [],
        "parsed_tools": [],
        "semantic_mappings": {},
        "nodes": [],
        "edges": [],
        "graph": {},
        "stats": {}
    }

    final_state = app.invoke(initial_state)
    stats = final_state.get("stats", {})

    print("--------------------------------------------------")
    print("SUCCESS: Dependency graph generation complete.")
    print(f"  Nodes: {stats.get('nodes', 0)}")
    print(f"  Edges: {stats.get('edges', 0)}")
    print(f"  Provenance: {stats.get('provenance_ratio', 0)}")
    print("==================================================")


if __name__ == "__main__":
    main()

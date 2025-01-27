import os
from openai import OpenAI
import networkx as nx
import json

############################################################
# 1. SETUP: Initialize OpenAI client
############################################################

# Initialize the client properly
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
)

############################################################
# 2. BUILD KNOWLEDGE GRAPH
############################################################

def build_knowledge_graph(text_data: str) -> nx.DiGraph:
    """
    Uses ChatGPT to extract entities and relationships from text,
    then builds a directed graph (nx.DiGraph).
    """
    # Step A: Use ChatGPT to parse the text into structured relationships
    system_prompt = (
        "As an expert information extraction system, analyze the provided text to identify "
        "entities and their interrelationships. Follow these guidelines:\n\n"
        "1. Entity Identification:\n"
        "- Extract all concrete entities (people, organizations, locations, concepts)\n"
        "- Use canonical names (e.g., 'ACME Corp' not 'the company')\n\n"
        "2. Relationship Extraction:\n"
        "- Identify specific verb-based relationships between entities\n"
        "- Use active voice (e.g., 'manufactures' not 'is manufacturer of')\n\n"
        "3. Output Requirements:\n"
        "- Maintain consistent JSON structure\n"
        "- Use double quotes for all JSON keys and values\n"
        "- Ensure entities appear exactly as named in text\n"
        "- Return only valid JSON with no additional formatting\n\n"
        "Output format:\n"
        "{\n"
        "  \"entities\": [\"Entity1\", \"Entity2\"],\n"
        "  \"relationships\": [\n"
        "    {\"source\": \"Entity1\", \"target\": \"Entity2\", \"relation\": \"verb phrase\"}\n"
        "  ]\n"
        "}"
    )
    
    user_prompt = f"Text to parse:\n{text_data}\n\nExtract entities and relationships."

    # Updated API call with response_format to ensure JSON
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    
    # Get response content and parse JSON
    raw_response = response.choices[0].message.content.strip()
    
    try:
        extraction = json.loads(raw_response)
    except json.JSONDecodeError:
        print("Error parsing ChatGPT response as JSON. Raw output:")
        print(raw_response)
        return nx.DiGraph()
    
    # Build a directed graph
    G = nx.DiGraph()
    
    # Add entities as nodes
    for entity in extraction.get("entities", []):
        G.add_node(entity)
    
    # Add relationships as edges
    for rel in extraction.get("relationships", []):
        source = rel.get("source")
        target = rel.get("target")
        relation = rel.get("relation", "")
        if source and target:
            G.add_edge(source, target, relation=relation)
    
    return G


############################################################
# 3. RETRIEVE RELEVANT KNOWLEDGE
############################################################

def get_relevant_knowledge(query: str, knowledge_graph: nx.DiGraph) -> str:
    """
    A simple 'retrieval' approach. We look for nodes and edges that match
    keywords from the query. For demonstration, we do a naive substring match.
    """
    # Extract keywords (very naive). In practice, you might use an NLP approach.
    query_lower = query.lower().split()
    
    relevant_nodes = set()
    
    # Check each node for any keyword match
    for node in knowledge_graph.nodes():
        for word in query_lower:
            if word in node.lower():
                relevant_nodes.add(node)
    
    # We expand edges that connect relevant nodes
    relevant_edges_info = []
    for source, target, data in knowledge_graph.edges(data=True):
        if source in relevant_nodes or target in relevant_nodes:
            relevant_edges_info.append(f"{source} --({data.get('relation','')})--> {target}")
    
    # Combine relevant knowledge into a textual summary
    summary = []
    if relevant_nodes:
        summary.append("Relevant Entities:\n" + ", ".join(relevant_nodes))
    if relevant_edges_info:
        summary.append("\nRelevant Relationships:\n" + "\n".join(relevant_edges_info))
    
    return "\n".join(summary)


############################################################
# 4. GENERATE ANSWER WITH CONTEXT
############################################################

def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer by passing the user query and relevant context
    into the ChatGPT model.
    """
    system_prompt = (
        "Analyze relationships CAREFULLY. Pay special attention to:"
        "\n- Relationship reasons (after '-->' in edges)"
        "\n- Indirect connections through multiple nodes"
        "\n- Ethical implications mentioned in relationship metadata"
        "\n\nReturn your answer as a JSON object with an 'answer' key."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Knowledge Graph Context:\n{context}\n\nUser Query:\n{query}"},
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("answer", "No answer found in response")
    except json.JSONDecodeError:
        return "Error: Failed to parse response as JSON"


############################################################
# 5. DEMO USAGE
############################################################

if __name__ == "__main__":
    # Sample text with nested relationships
    sample_text = """
    Vertex Dynamics, a Boston-based AI startup, recently acquired two companies: 
    1) NeuroSynth Analytics for $800 million, gaining their real-time neural network optimization platform
    2) QuantumCore Systems for $1.2 billion, obtaining their quantum machine learning patents
    
    The NeuroSynth acquisition included their lead product CortexOptimize, which utilizes QuantumCore's 
    QML-9 algorithms for energy-efficient AI processing. Following these mergers, Vertex's CTO Dr. Lena Marquez 
    announced a partnership with Singapore-based Horizon Technologies to develop hybrid quantum-AI chipsets.
    
    Meanwhile, Horizon Technologies revealed their new neural processor design leverages both CortexOptimize's 
    architecture and QuantumCore's latest entanglement compression techniques. This development has drawn 
    interest from the EU's AI Ethics Board due to potential dual-use applications.
    
    In unrelated news, Vertex's stock rose 7% after announcing a collaboration with Shanghai AI Lab on 
    biomedical neural interfaces.
    """
    
    # Build the knowledge graph
    kg = build_knowledge_graph(sample_text)

    # Multi-hop query requiring graph traversal
    user_query = "What ethical concerns have been raised about technologies derived from Vertex's quantum computing acquisitions?"

    # Retrieve relevant knowledge from the graph
    retrieved_knowledge = get_relevant_knowledge(user_query, kg)
    print("=== Retrieved Knowledge ===")
    print(retrieved_knowledge)
    print("===========================")

    # Generate final answer
    final_answer = generate_answer(user_query, retrieved_knowledge)
    print("=== Answer ===")
    print(final_answer)

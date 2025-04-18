"""
Infomap algorithm for community detection.
"""

import infomap
import networkx as nx
import time
from collections import defaultdict

def detect_communities(graph):
    """
    Detect communities using the Infomap algorithm.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    
    Returns:
    --------
    dict
        Dictionary mapping community ID to list of nodes
    float
        Execution time in seconds
    """
    start_time = time.time()
    
    # Set up Infomap
    im = infomap.Infomap("--two-level")
    
    # Add links to Infomap
    for source, target in graph.edges():
        im.add_link(source, target)
    
    # Run Infomap
    im.run()
    
    # Extract communities
    communities = defaultdict(list)
    for node in im.tree:
        if node.is_leaf:
            communities[node.module_id].append(node.node_id)
    
    execution_time = time.time() - start_time
    
    print(f"Infomap found {len(communities)} communities in {execution_time:.2f} seconds")
    
    return dict(communities), execution_time

def handle_boundary_nodes(graph, communities):
    """
    Use Infomap to reassign boundary nodes to the most appropriate community.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    dict
        Dictionary with refined community assignments
    """
    # Identify boundary nodes (nodes that have neighbors in different communities)
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = comm_id
    
    boundary_nodes = set()
    for node in graph.nodes():
        if node in node_to_community:
            node_comm = node_to_community[node]
            for neighbor in graph.neighbors(node):
                if neighbor in node_to_community and node_to_community[neighbor] != node_comm:
                    boundary_nodes.add(node)
                    break
    
    if not boundary_nodes:
        print("No boundary nodes found. Communities are completely separate.")
        return communities
    
    print(f"Found {len(boundary_nodes)} boundary nodes.")
    
    # Create a subgraph of boundary nodes and their neighbors
    boundary_neighborhood = set(boundary_nodes)
    for node in boundary_nodes:
        boundary_neighborhood.update(graph.neighbors(node))
    
    boundary_graph = graph.subgraph(boundary_neighborhood).copy()
    
    # Apply Infomap to the boundary subgraph
    im = infomap.Infomap("--two-level")
    
    # Add links
    for source, target in boundary_graph.edges():
        im.add_link(source, target)
    
    # Run Infomap
    im.run()
    
    # Get the results
    infomap_communities = defaultdict(list)
    for node in im.tree:
        if node.is_leaf:
            infomap_communities[node.module_id].append(node.node_id)
    
    # Update community assignments for boundary nodes
    refined_communities = {k: list(v) for k, v in communities.items()}
    next_comm_id = max(refined_communities.keys()) + 1 if refined_communities else 0
    
    for module_id, nodes in infomap_communities.items():
        # Check if this is a new community
        is_new_community = True
        for existing_comm_id, existing_nodes in refined_communities.items():
            overlap = set(nodes) & set(existing_nodes)
            if len(overlap) > len(nodes) / 2:
                is_new_community = False
                break
        
        if is_new_community:
            # Only include boundary nodes in the new community
            boundary_members = [n for n in nodes if n in boundary_nodes]
            if len(boundary_members) > 0:
                # Remove these nodes from their original communities
                for node in boundary_members:
                    for comm_id in list(refined_communities.keys()):
                        if node in refined_communities[comm_id]:
                            refined_communities[comm_id].remove(node)
                
                # Add them to the new community
                refined_communities[next_comm_id] = boundary_members
                next_comm_id += 1
    
    # Clean up empty communities
    refined_communities = {k: v for k, v in refined_communities.items() if v}
    
    print(f"After boundary handling: {len(refined_communities)} communities")
    
    return refined_communities
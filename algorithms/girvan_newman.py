"""
Girvan-Newman algorithm for community detection.
"""

import networkx as nx
import time
import numpy as np

def detect_communities(graph, max_iterations=None):
    """
    Detect communities using the Girvan-Newman algorithm.
    Note: This is computationally expensive for large graphs.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    max_iterations : int, optional
        Maximum number of edge removals to perform
    
    Returns:
    --------
    dict
        Dictionary mapping community ID to list of nodes
    float
        Execution time in seconds
    """
    start_time = time.time()
    
    if graph.number_of_nodes() > 1000:
        print("Warning: Girvan-Newman algorithm is computationally expensive for graphs with >1000 nodes.")
        print("Using a modified approach for large graphs.")
        communities, execution_time = _detect_communities_large_graph(graph, max_iterations)
        return communities, execution_time
    
    # For small graphs, use the standard algorithm
    # If max_iterations is not specified, limit based on graph size
    if max_iterations is None:
        max_iterations = min(int(graph.number_of_edges() * 0.1), 100)
    
    # Run the algorithm
    communities_generator = nx.algorithms.community.girvan_newman(graph)
    
    # Get communities from the generator
    communities = None
    best_modularity = -1
    
    # Take the first few iterations and select the one with highest modularity
    for i in range(max_iterations):
        try:
            comm_tuple = next(communities_generator)
            # Convert to list of lists for easier handling
            comm_list = [list(c) for c in comm_tuple]
            
            # Calculate modularity
            modularity = nx.algorithms.community.modularity(graph, comm_tuple)
            
            if modularity > best_modularity:
                best_modularity = modularity
                communities = comm_list
                
        except StopIteration:
            break
    
    # Convert to dict format
    communities_dict = {i: nodes for i, nodes in enumerate(communities)}
    
    execution_time = time.time() - start_time
    
    print(f"Girvan-Newman found {len(communities_dict)} communities in {execution_time:.2f} seconds")
    
    return communities_dict, execution_time

def _detect_communities_large_graph(graph, max_iterations=None):
    """
    Modified Girvan-Newman approach for large graphs.
    Calculates edge betweenness only once and removes multiple edges.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    max_iterations : int, optional
        Maximum number of edge removals to perform
    
    Returns:
    --------
    dict
        Dictionary mapping community ID to list of nodes
    float
        Execution time in seconds
    """
    start_time = time.time()
    
    # Work with a copy of the graph
    G = graph.copy()
    
    # If max_iterations is not specified, use a reasonable default
    if max_iterations is None:
        max_iterations = min(int(G.number_of_edges() * 0.05), 50)
    
    # Calculate edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # Sort edges by betweenness (highest first)
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
    
    # Remove the top edges
    edges_to_remove = min(max_iterations, len(sorted_edges))
    
    for i in range(edges_to_remove):
        edge = sorted_edges[i][0]
        if G.has_edge(*edge):
            G.remove_edge(*edge)
    
    # Find connected components (communities)
    components = list(nx.connected_components(G))
    communities_dict = {i: list(component) for i, component in enumerate(components)}
    
    execution_time = time.time() - start_time
    
    print(f"Modified Girvan-Newman found {len(communities_dict)} communities in {execution_time:.2f} seconds")
    
    return communities_dict, execution_time

def refine_communities(graph, communities, max_edges_per_community=None):
    """
    Refine communities using the Girvan-Newman principle of edge betweenness.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    max_edges_per_community : int, optional
        Maximum number of edges to remove per community
    
    Returns:
    --------
    dict
        Dictionary mapping community ID to list of nodes
    """
    refined_communities = {}
    community_id_counter = 0
    
    for comm_id, nodes in communities.items():
        # Skip tiny communities
        if len(nodes) < 5:
            refined_communities[community_id_counter] = nodes
            community_id_counter += 1
            continue
        
        # Extract the subgraph for this community
        subgraph = graph.subgraph(nodes).copy()
        
        # Determine max edges to remove (proportional to subgraph size)
        if max_edges_per_community is None:
            max_edges = max(1, int(subgraph.number_of_edges() * 0.1))
        else:
            max_edges = max_edges_per_community
        
        # Calculate edge betweenness for the subgraph
        try:
            edge_betweenness = nx.edge_betweenness_centrality(subgraph)
            
            # Sort edges by betweenness (highest first)
            sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
            
            # Remove top edges
            edges_to_remove = min(max_edges, len(sorted_edges))
            temp_graph = subgraph.copy()
            
            for i in range(edges_to_remove):
                edge = sorted_edges[i][0]
                if temp_graph.has_edge(*edge):
                    temp_graph.remove_edge(*edge)
            
            # Find connected components
            components = list(nx.connected_components(temp_graph))
            
            # If we've split the community, use the new components
            if len(components) > 1:
                for component in components:
                    refined_communities[community_id_counter] = list(component)
                    community_id_counter += 1
            else:
                # If we couldn't split it, keep the original community
                refined_communities[community_id_counter] = nodes
                community_id_counter += 1
                
        except Exception as e:
            print(f"Error refining community {comm_id}: {e}")
            # Keep the original community
            refined_communities[community_id_counter] = nodes
            community_id_counter += 1
    
    return refined_communities
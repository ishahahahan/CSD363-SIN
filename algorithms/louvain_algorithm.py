"""
Louvain method for community detection.
"""

import networkx as nx
import community as community_louvain  # python-louvain package
import time
from collections import defaultdict

def detect_communities(graph):
    """
    Detect communities using the Louvain method.
    
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
    
    # Apply Louvain algorithm
    partition = community_louvain.best_partition(graph)
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    execution_time = time.time() - start_time
    
    print(f"Louvain method found {len(communities)} communities in {execution_time:.2f} seconds")
    
    return dict(communities), execution_time

def calculate_modularity(graph, communities):
    """
    Calculate modularity of a community partition.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    float
        Modularity score
    """
    # Convert communities dict to the format expected by networkx
    comm_sets = [set(nodes) for nodes in communities.values()]
    
    # Calculate modularity
    modularity = nx.algorithms.community.modularity(graph, comm_sets)
    return modularity
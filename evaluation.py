import time
import logging
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

def compute_modularity(G, partition):
    """
    Compute modularity for a given graph and partition.
    
    Args:
        G (networkx.Graph): Input graph
        partition (dict): Mapping of node to community ID
    
    Returns:
        float: Modularity score
    """
    return community_louvain.modularity(partition, G)

def compute_conductance(G, communities):
    """
    Compute conductance for each community and average conductance.
    
    Conductance is defined as cut_size/volume where:
    - cut_size: number of edges between the community and the rest of the graph
    - volume: total number of edges connected to nodes in the community
    
    Lower conductance values indicate better communities.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to list of nodes
    
    Returns:
        tuple: (list of conductance values, average conductance)
    """
    conductance_values = []
    
    for comm_id, nodes in communities.items():
        if not nodes:  # Skip empty communities
            continue
            
        # Create set for faster lookups
        community_set = set(nodes)
        
        # Calculate cut size (edges between community and rest of graph)
        cut_size = 0
        # Calculate volume (total degree of nodes in community)
        volume = 0
        
        for node in community_set:
            if node not in G:  # Skip nodes not in graph
                continue
                
            neighbors = set(G.neighbors(node))
            # Degree of current node
            node_degree = len(neighbors)
            volume += node_degree
            
            # Count edges to nodes outside community
            outside_edges = len(neighbors - community_set)
            cut_size += outside_edges
        
        # Handle edge cases
        if volume == 0:
            conductance = 1.0  # Worst case
        else:
            conductance = cut_size / volume
            
        conductance_values.append(conductance)
    
    # Compute average conductance
    avg_conductance = np.mean(conductance_values) if conductance_values else 0.0
    
    return conductance_values, avg_conductance

def compute_nmi(partition, ground_truth):
    """
    Compute Normalized Mutual Information between detected communities and ground truth.
    
    Args:
        partition (dict): Mapping of node to detected community ID
        ground_truth (dict): Mapping of node to ground truth community ID
    
    Returns:
        float: NMI score
    """
    if not ground_truth:
        logger.warning("No ground truth provided for NMI calculation")
        return None
        
    # Find common nodes
    common_nodes = set(partition.keys()) & set(ground_truth.keys())
    
    if not common_nodes:
        logger.warning("No common nodes between partition and ground truth")
        return 0.0
        
    # Extract community assignments for common nodes
    true_labels = [ground_truth[node] for node in common_nodes]
    pred_labels = [partition[node] for node in common_nodes]
    
    # Compute NMI
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return nmi

def evaluate_all(G, final_partition, ground_truth=None):
    """
    Evaluate the final partition using multiple metrics.
    
    Args:
        G (networkx.Graph): Input graph
        final_partition (dict): Mapping of node to community ID
        ground_truth (dict, optional): Ground truth community assignments
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info("Evaluating community detection results")
    start_time = time.time()
    
    # Create communities dict
    communities = defaultdict(list)
    for node, comm_id in final_partition.items():
        communities[comm_id].append(node)
    
    # Compute modularity
    modularity = compute_modularity(G, final_partition)
    logger.info(f"Modularity: {modularity:.4f}")
    
    # Compute conductance
    conductance_values, avg_conductance = compute_conductance(G, communities)
    logger.info(f"Average conductance: {avg_conductance:.4f}")
    
    # Compute NMI if ground truth is provided
    nmi = None
    if ground_truth:
        nmi = compute_nmi(final_partition, ground_truth)
        logger.info(f"Normalized Mutual Information: {nmi:.4f}")
    else:
        logger.info("No ground truth provided, skipping NMI calculation")
    
    # Create metrics dictionary
    metrics = {
        "modularity": modularity,
        "conductance_values": conductance_values,
        "avg_conductance": avg_conductance,
        "nmi": nmi,
        "num_communities": len(communities),
        "community_sizes": {comm_id: len(nodes) for comm_id, nodes in communities.items()}
    }
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f} seconds")
    
    if elapsed > 60:
        logger.warning(f"Evaluation took {elapsed:.2f} seconds, which exceeds the 60-second threshold")
    
    return metrics

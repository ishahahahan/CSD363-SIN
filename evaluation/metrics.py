"""
Evaluation metrics for community detection algorithms.
"""

import networkx as nx
import numpy as np
import math
from collections import Counter
from sklearn import metrics as sk_metrics

def modularity(graph, communities):
    """
    Calculate modularity score for a community partition.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    float
        Modularity score (-0.5 to 1.0, higher is better)
    """
    # Convert communities dict to the format expected by networkx (list of sets)
    comm_sets = [set(nodes) for nodes in communities.values()]
    
    try:
        # Calculate modularity using networkx implementation
        mod_score = nx.algorithms.community.modularity(graph, comm_sets)
        return mod_score
    except Exception as e:
        print(f"Error calculating modularity: {e}")
        return float('nan')

def normalized_mutual_information(communities, ground_truth):
    """
    Calculate Normalized Mutual Information between detected communities and ground truth.
    
    Parameters:
    -----------
    communities : dict
        Dictionary mapping community ID to list of nodes
    ground_truth : dict
        Dictionary mapping node ID to ground truth community ID
    
    Returns:
    --------
    float
        NMI score (0 to 1, higher is better)
    """
    # Check if we have ground truth labels
    if not ground_truth:
        print("No ground truth data available for NMI calculation.")
        return float('nan')
    
    # Create predicted labels
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = comm_id
    
    # Filter to include only nodes that have ground truth and are in the communities
    common_nodes = set(node_to_community.keys()) & set(ground_truth.keys())
    
    if not common_nodes:
        print("No overlap between predicted communities and ground truth.")
        return 0.0
    
    # Create label arrays
    y_true = []
    y_pred = []
    
    for node in common_nodes:
        y_true.append(ground_truth[node])
        y_pred.append(node_to_community[node])
    
    # Calculate NMI
    try:
        nmi_score = sk_metrics.normalized_mutual_info_score(y_true, y_pred)
        return nmi_score
    except Exception as e:
        print(f"Error calculating NMI: {e}")
        return float('nan')

def conductance(graph, communities):
    """
    Calculate conductance for each community and return the average.
    Conductance measures the fraction of total edge volume that points outside the community.
    Lower conductance means better-defined communities.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    float
        Average conductance (0 to 1, lower is better)
    dict
        Conductance for each community
    """
    conductance_scores = {}
    
    for comm_id, nodes in communities.items():
        # Skip empty communities
        if not nodes:
            continue
            
        nodes_set = set(nodes)
        
        # Count internal and external edges
        internal_edges = 0
        external_edges = 0
        
        for node in nodes:
            if node not in graph:
                continue
                
            for neighbor in graph.neighbors(node):
                if neighbor in nodes_set:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Divide by 2 because each internal edge is counted twice
        internal_edges = internal_edges / 2
        
        # Calculate conductance
        if internal_edges + external_edges > 0:
            conductance_scores[comm_id] = external_edges / (2 * internal_edges + external_edges)
        else:
            conductance_scores[comm_id] = 0.0
    
    # Calculate average conductance
    if conductance_scores:
        avg_conductance = sum(conductance_scores.values()) / len(conductance_scores)
    else:
        avg_conductance = float('nan')
    
    return avg_conductance, conductance_scores

def adjusted_rand_index(communities, ground_truth):
    """
    Calculate Adjusted Rand Index between detected communities and ground truth.
    
    Parameters:
    -----------
    communities : dict
        Dictionary mapping community ID to list of nodes
    ground_truth : dict
        Dictionary mapping node ID to ground truth community ID
    
    Returns:
    --------
    float
        Adjusted Rand Index score (-0.5 to 1.0, higher is better)
    """
    # Check if we have ground truth labels
    if not ground_truth:
        print("No ground truth data available for ARI calculation.")
        return float('nan')
    
    # Create predicted labels
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = comm_id
    
    # Filter to include only nodes that have ground truth and are in the communities
    common_nodes = set(node_to_community.keys()) & set(ground_truth.keys())
    
    if not common_nodes:
        print("No overlap between predicted communities and ground truth.")
        return -0.5
    
    # Create label arrays
    y_true = []
    y_pred = []
    
    for node in common_nodes:
        y_true.append(ground_truth[node])
        y_pred.append(node_to_community[node])
    
    # Calculate ARI
    try:
        ari_score = sk_metrics.adjusted_rand_score(y_true, y_pred)
        return ari_score
    except Exception as e:
        print(f"Error calculating ARI: {e}")
        return float('nan')

def community_size_stats(communities):
    """
    Calculate statistics about community sizes.
    
    Parameters:
    -----------
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    dict
        Dictionary containing community size statistics
    """
    sizes = [len(nodes) for nodes in communities.values()]
    
    if not sizes:
        return {
            "min_size": 0,
            "max_size": 0,
            "avg_size": 0,
            "median_size": 0,
            "total_communities": 0,
            "size_distribution": {}
        }
    
    # Calculate size distribution
    size_counts = Counter(sizes)
    
    return {
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": sum(sizes) / len(sizes),
        "median_size": sorted(sizes)[len(sizes) // 2],
        "total_communities": len(sizes),
        "size_distribution": dict(size_counts)
    }

def coverage(graph, communities):
    """
    Calculate coverage - the fraction of intra-community edges.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    float
        Coverage (0 to 1, higher is better)
    """
    # Create a mapping from node to community ID
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = comm_id
    
    # Count intra-community edges
    intra_edges = 0
    total_edges = graph.number_of_edges()
    
    for edge in graph.edges():
        source, target = edge
        if (source in node_to_community and 
            target in node_to_community and
            node_to_community[source] == node_to_community[target]):
            intra_edges += 1
    
    if total_edges > 0:
        return intra_edges / total_edges
    else:
        return 0.0

def evaluate_all(graph, communities, ground_truth=None):
    """
    Evaluate a community detection result using multiple metrics.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    ground_truth : dict, optional
        Dictionary mapping node ID to ground truth community ID
    
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    mod = modularity(graph, communities)
    avg_conductance, conductance_scores = conductance(graph, communities)
    cov = coverage(graph, communities)
    size_stats = community_size_stats(communities)
    
    results = {
        "modularity": mod,
        "avg_conductance": avg_conductance,
        "coverage": cov,
        "community_stats": size_stats
    }
    
    # Add metrics that require ground truth
    if ground_truth:
        nmi = normalized_mutual_information(communities, ground_truth)
        ari = adjusted_rand_index(communities, ground_truth)
        results["nmi"] = nmi
        results["adjusted_rand_index"] = ari
    
    return results

def print_evaluation_summary(evaluation_results):
    """
    Print a summary of the evaluation results.
    
    Parameters:
    -----------
    evaluation_results : dict
        Dictionary containing evaluation metrics
    """
    print("\n===== Community Detection Evaluation =====")
    print(f"Modularity: {evaluation_results['modularity']:.4f}")
    print(f"Average Conductance: {evaluation_results['avg_conductance']:.4f} (lower is better)")
    print(f"Coverage: {evaluation_results['coverage']:.4f}")
    
    if "nmi" in evaluation_results:
        print(f"Normalized Mutual Information: {evaluation_results['nmi']:.4f}")
    
    if "adjusted_rand_index" in evaluation_results:
        print(f"Adjusted Rand Index: {evaluation_results['adjusted_rand_index']:.4f}")
    
    # Community statistics
    stats = evaluation_results["community_stats"]
    print(f"\nTotal Communities: {stats['total_communities']}")
    print(f"Community Size - Min: {stats['min_size']}, Max: {stats['max_size']}, Average: {stats['avg_size']:.1f}")
    
    # Show size distribution in a compact way
    print("\nCommunity Size Distribution:")
    ranges = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    
    for start, end in ranges:
        count = sum(stats["size_distribution"].get(size, 0) for size in range(start, int(min(end, stats["max_size"])) + 1))
        if end == float('inf'):
            print(f"  > {start}: {count} communities")
        else:
            print(f"  {start}-{end}: {count} communities")
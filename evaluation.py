import time
import logging
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from collections import defaultdict

logger = logging.getLogger('community_pipeline')

def compute_modularity(G, partition):
    """
    Compute modularity for a given graph and partition.
    
    Args:
        G (networkx.Graph): Input graph
        partition (dict): Mapping of node to community ID
    
    Returns:
        float: Modularity score
    """
    try:
        # Filter the partition to include only nodes in G
        filtered_partition = {node: comm for node, comm in partition.items() if node in G}
        
        if not filtered_partition:
            logger.warning("No valid nodes in partition after filtering")
            return 0.0
            
        # Calculate modularity using the built-in function
        modularity = community_louvain.modularity(filtered_partition, G)
        
        # Sanity check - modularity should typically be between -0.5 and 1.0
        if not (-0.5 <= modularity <= 1.0):
            logger.warning(f"Suspicious modularity value: {modularity}. Expected between -0.5 and 1.0.")
            
        return modularity
    except Exception as e:
        logger.error(f"Error computing modularity: {str(e)}")
        return 0.0  # Return 0 instead of None to maintain numeric consistency in pipeline

def compute_conductance(G, communities):
    """
    Compute conductance for each community and average conductance.
    
    Conductance is defined as cut_size/min(vol(S), vol(V-S)) where:
    - cut_size: number of edges between the community and the rest of the graph
    - vol(S): sum of degrees of nodes in the community
    - vol(V-S): sum of degrees of nodes outside the community
    
    Lower conductance values indicate better communities.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to list of nodes
    
    Returns:
        tuple: (list of conductance values, average conductance)
    """
    conductance_values = []
    community_conductances = {}  # Store individual community conductances
    
    # For empty graph
    if G.number_of_edges() == 0:
        logger.warning("Graph has no edges, conductance calculation not possible")
        return [], 0.0
    
    # Total volume of the graph (sum of all degrees)
    total_volume = sum(dict(G.degree()).values())
    
    for comm_id, nodes in communities.items():
        if not nodes:  # Skip empty communities
            continue
            
        # Filter nodes to only those in the graph
        valid_nodes = [node for node in nodes if node in G]
        
        if not valid_nodes:
            logger.debug(f"Community {comm_id} has no nodes in the graph, skipping")
            continue
        
        # Calculate the cut size (edges leaving the community)
        cut_size = 0
        
        # Calculate volume of community (sum of degrees of nodes in community)
        community_volume = 0
        
        # Create a set of community nodes for faster lookups
        community_set = set(valid_nodes)
        
        for node in valid_nodes:
            degree = G.degree(node)
            community_volume += degree
            
            # Count edges that cross community boundary
            for neighbor in G.neighbors(node):
                if neighbor not in community_set:
                    cut_size += 1
        
        # Calculate the volume of the rest of the graph
        rest_volume = total_volume - community_volume
        
        # Calculate conductance
        if min(community_volume, rest_volume) > 0:
            conductance = cut_size / min(community_volume, rest_volume)
        else:
            if cut_size == 0:  # Isolated community or single node
                conductance = 0.0  # Perfect conductance
            else:
                conductance = 1.0  # Worst conductance
        
        # Ensure conductance is between 0 and 1
        conductance = max(0.0, min(1.0, conductance))
        
        conductance_values.append(conductance)
        community_conductances[comm_id] = conductance
    
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
    
    try:
        # Compute NMI
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        return nmi
    except Exception as e:
        logger.error(f"Error computing NMI: {str(e)}")
        return 0.0

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
    
    # Validate inputs
    if G is None or not isinstance(G, nx.Graph):
        logger.error("Invalid graph object provided")
        return {
            "modularity": 0.0,
            "avg_conductance": 1.0,
            "conductance_values": [],
            "nmi": None,
            "num_communities": 0,
            "community_sizes": {},
            "error": "Invalid graph object"
        }
    
    if not final_partition:
        logger.error("Empty partition provided")
        return {
            "modularity": 0.0,
            "avg_conductance": 1.0,
            "conductance_values": [],
            "nmi": None,
            "num_communities": 0,
            "community_sizes": {},
            "error": "Empty partition"
        }
    
    # Create communities dict
    communities = defaultdict(list)
    for node, comm_id in final_partition.items():
        communities[comm_id].append(node)
    
    # Basic statistics
    num_communities = len(communities)
    community_sizes = {comm_id: len(nodes) for comm_id, nodes in communities.items()}
    
    # Calculate size statistics
    sizes = list(community_sizes.values())
    size_stats = {
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "mean": float(np.mean(sizes)) if sizes else 0,
        "median": float(np.median(sizes)) if sizes else 0,
        "std": float(np.std(sizes)) if sizes else 0
    }
    
    # Check for nodes in partition not in graph
    nodes_in_graph = set(G.nodes())
    nodes_in_partition = set(final_partition.keys())
    missing_nodes = nodes_in_partition - nodes_in_graph
    
    if missing_nodes:
        logger.warning(f"{len(missing_nodes)} nodes in partition not found in graph")
    
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
    
    # Calculate coverage (fraction of edges within communities)
    intra_edges = 0
    total_edges = G.number_of_edges()
    
    for comm_id, nodes in communities.items():
        # Filter to nodes in graph
        valid_nodes = set(nodes) & nodes_in_graph
        if len(valid_nodes) > 1:  # Need at least 2 nodes for an edge
            subgraph = G.subgraph(valid_nodes)
            intra_edges += subgraph.number_of_edges()
    
    coverage = intra_edges / total_edges if total_edges > 0 else 0
    
    # Calculate average degree within communities
    avg_internal_degree = 2 * intra_edges / sum(len(nodes) for nodes in communities.values()) if communities else 0
    
    # Create metrics dictionary
    metrics = {
        "modularity": modularity,
        "conductance_values": conductance_values,
        "avg_conductance": avg_conductance,
        "nmi": nmi,
        "num_communities": num_communities,
        "community_sizes": community_sizes,
        "size_stats": size_stats,
        "coverage": coverage,
        "avg_internal_degree": avg_internal_degree,
        "missing_nodes": len(missing_nodes),
        "evaluation_time": time.time() - start_time
    }
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f} seconds")
    
    return metrics

def compare_algorithms(G, algorithm_results, ground_truth=None):
    """
    Compare results from multiple community detection algorithms.
    
    Args:
        G (networkx.Graph): Input graph
        algorithm_results (dict): Dict mapping algorithm names to their partitions
        ground_truth (dict, optional): Ground truth community assignments
    
    Returns:
        dict: Dictionary comparing metrics across algorithms
    """
    logger.info(f"Comparing {len(algorithm_results)} community detection algorithms")
    
    comparison = {}
    
    for algo_name, partition in algorithm_results.items():
        logger.info(f"Evaluating algorithm: {algo_name}")
        metrics = evaluate_all(G, partition, ground_truth)
        comparison[algo_name] = metrics
    
    # Create summary table
    summary = {
        "algorithm": [],
        "modularity": [],
        "avg_conductance": [],
        "num_communities": [],
        "coverage": []
    }
    
    if any(metrics.get('nmi') is not None for metrics in comparison.values()):
        summary["nmi"] = []
    
    for algo_name, metrics in comparison.items():
        summary["algorithm"].append(algo_name)
        summary["modularity"].append(metrics["modularity"])
        summary["avg_conductance"].append(metrics["avg_conductance"])
        summary["num_communities"].append(metrics["num_communities"])
        summary["coverage"].append(metrics["coverage"])
        
        if "nmi" in summary and metrics.get("nmi") is not None:
            summary["nmi"].append(metrics["nmi"])
    
    # Calculate best algorithm for each metric
    best_modularity = max(summary["modularity"])
    best_modularity_idx = summary["modularity"].index(best_modularity)
    best_modularity_algo = summary["algorithm"][best_modularity_idx]
    
    best_conductance = min(summary["avg_conductance"])
    best_conductance_idx = summary["avg_conductance"].index(best_conductance)
    best_conductance_algo = summary["algorithm"][best_conductance_idx]
    
    summary["best"] = {
        "modularity": {
            "value": best_modularity,
            "algorithm": best_modularity_algo
        },
        "conductance": {
            "value": best_conductance,
            "algorithm": best_conductance_algo
        }
    }
    
    # Add the summary to the comparison results
    comparison["summary"] = summary
    
    return comparison

def calculate_improvement(metrics1, metrics2):
    """
    Calculate improvement between two sets of metrics.
    
    Args:
        metrics1 (dict): First set of metrics (baseline)
        metrics2 (dict): Second set of metrics (improved)
    
    Returns:
        dict: Dictionary of improvement metrics
    """
    return {
        'modularity': metrics2['modularity'] - metrics1['modularity'],
        'conductance': metrics1['avg_conductance'] - metrics2['avg_conductance'],
        'num_communities': metrics2['num_communities'] - metrics1['num_communities'],
        'coverage': metrics2['coverage'] - metrics1['coverage'] if 'coverage' in metrics1 and 'coverage' in metrics2 else None
    }
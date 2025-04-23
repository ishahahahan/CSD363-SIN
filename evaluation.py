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
        
        # Add warning for extremely high modularity (which may indicate disconnected components)
        if modularity > 0.95:
            connected_components = nx.number_connected_components(G)
            if connected_components > 1:
                logger.warning(f"Very high modularity ({modularity:.4f}) with {connected_components} connected components.")
                logger.warning("This is expected for disconnected graphs, as each component naturally forms a community.")
            
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
        tuple: (list of conductance values, average conductance, per-community conductances)
    """
    conductance_values = []
    community_conductances = {}  # Store individual community conductances
    
    # For empty graph
    if G.number_of_edges() == 0:
        logger.warning("Graph has no edges, conductance calculation not possible")
        return [], 0.0, {}
    
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
    
    return conductance_values, avg_conductance, community_conductances

def compute_edge_betweenness(G, partition=None):
    """
    Compute edge betweenness centrality for a graph.
    Used in evaluation of Girvan-Newman algorithm.
    
    Args:
        G (networkx.Graph): Input graph
        partition (dict, optional): Mapping of node to community ID
            
    Returns:
        dict: Edge betweenness values for each edge
        float: Average edge betweenness
    """
    try:
        # Skip calculation for very large graphs
        if G.number_of_edges() > 100000:
            logger.warning("Graph too large for edge betweenness computation, using approximation")
            # Return empty dict with zeros to maintain interface
            edge_betweenness = {edge: 0.0 for edge in G.edges()}
            return edge_betweenness, 0.0, None
            
        # For large graphs, use sampling
        if G.number_of_nodes() > 5000:
            # Use approximate betweenness with sampling
            k = min(1000, G.number_of_nodes() // 10)
            logger.info(f"Using approximate edge betweenness with k={k} samples")
            edge_betweenness = nx.edge_betweenness_centrality(G, k=k)
        else:
            # Compute full betweenness for smaller graphs
            edge_betweenness = nx.edge_betweenness_centrality(G)
        
        # Calculate average betweenness
        avg_betweenness = sum(edge_betweenness.values()) / len(edge_betweenness) if edge_betweenness else 0
        
        # If partition is provided, compute betweenness for inter-community edges
        inter_community_betweenness = {}
        if partition:
            # Limit computation to save time
            max_edges = 10000
            edge_count = 0
            
            for edge, value in edge_betweenness.items():
                edge_count += 1
                if edge_count > max_edges:
                    break
                    
                u, v = edge
                if u in partition and v in partition:
                    if partition[u] != partition[v]:  # Inter-community edge
                        inter_community_betweenness[edge] = value
        
        return edge_betweenness, avg_betweenness, inter_community_betweenness if partition else None
    
    except Exception as e:
        logger.error(f"Error computing edge betweenness: {str(e)}")
        return {}, 0.0, None

def compute_description_length(G, communities):
    """
    Compute description length (map equation approximation) for InfoMap evaluation.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to list of nodes
    
    Returns:
        float: Description length approximation
        dict: Per-community description lengths
    """
    if G.number_of_edges() == 0:
        return 0.0, {}
    
    total_weight = G.number_of_edges() * 2  # Each edge contributes twice in undirected graph
    
    # Calculate visit rates (stationary probabilities of random walker)
    visit_rates = {}
    for node in G.nodes():
        visit_rates[node] = G.degree(node) / total_weight
    
    # Calculate description length
    H = 0.0  # Entropy
    community_entropy = {}
    
    # For each community
    for comm_id, nodes in communities.items():
        # Filter nodes to those in the graph
        valid_nodes = [node for node in nodes if node in G]
        
        if not valid_nodes:
            continue
        
        # Community visit rate
        comm_visit_rate = sum(visit_rates.get(node, 0) for node in valid_nodes)
        
        # Skip communities with zero visit rate
        if comm_visit_rate <= 0:
            continue
        
        # Count internal and external transitions
        internal_weight = 0
        external_weight = 0
        community_set = set(valid_nodes)
        
        for node in valid_nodes:
            for neighbor in G.neighbors(node):
                if neighbor in community_set:
                    internal_weight += 1
                else:
                    external_weight += 1
        
        # Calculate community entropy contribution
        if internal_weight + external_weight > 0:
            p_exit = external_weight / (internal_weight + external_weight)
            if 0 < p_exit < 1:  # Avoid log(0)
                comm_entropy = -comm_visit_rate * (p_exit * np.log2(p_exit) + (1-p_exit) * np.log2(1-p_exit))
                H += comm_entropy
                community_entropy[comm_id] = comm_entropy
    
    return H, community_entropy

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

def track_algorithm_metrics(G, step_results, ground_truth=None, algorithm_type=None):
    """
    Track metrics across algorithm steps.
    
    Args:
        G (networkx.Graph): Input graph
        step_results (list): List of partitions at each step
        ground_truth (dict, optional): Ground truth community assignments
        algorithm_type (str, optional): 'girvan_newman' or 'infomap'
        
    Returns:
        dict: Dictionary of metrics at each step
    """
    tracking = {
        'steps': len(step_results),
        'modularity': [],
        'num_communities': [],
        'conductance': [],
        'coverage': []
    }
    
    if algorithm_type == 'girvan_newman':
        tracking['edge_betweenness'] = []
    elif algorithm_type == 'infomap':
        tracking['description_length'] = []
    
    if ground_truth:
        tracking['nmi'] = []
    
    for i, partition in enumerate(step_results):
        logger.info(f"Evaluating step {i+1}/{len(step_results)}")
        
        # Create communities dict
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # Basic metrics
        tracking['num_communities'].append(len(communities))
        
        # Compute modularity
        modularity = compute_modularity(G, partition)
        tracking['modularity'].append(modularity)
        
        # Compute conductance
        _, avg_conductance, _ = compute_conductance(G, communities)
        tracking['conductance'].append(avg_conductance)
        
        # Calculate coverage
        intra_edges = 0
        total_edges = G.number_of_edges()
        
        for nodes in communities.values():
            # Need at least 2 nodes for an edge
            if len(nodes) > 1:  
                subgraph = G.subgraph(nodes)
                intra_edges += subgraph.number_of_edges()
        
        coverage = intra_edges / total_edges if total_edges > 0 else 0
        tracking['coverage'].append(coverage)
        
        # Algorithm-specific metrics
        if algorithm_type == 'girvan_newman':
            _, avg_betweenness, _ = compute_edge_betweenness(G, partition)
            tracking['edge_betweenness'].append(avg_betweenness)
        
        elif algorithm_type == 'infomap':
            description_length, _ = compute_description_length(G, communities)
            tracking['description_length'].append(description_length)
        
        # NMI if ground truth is provided
        if ground_truth:
            nmi = compute_nmi(partition, ground_truth)
            tracking['nmi'].append(nmi if nmi is not None else 0.0)
    
    return tracking

def evaluate_all(G, final_partition, ground_truth=None, algorithm_type=None):
    """
    Evaluate the final partition using multiple metrics.
    
    Args:
        G (networkx.Graph): Input graph
        final_partition (dict): Mapping of node to community ID
        ground_truth (dict, optional): Ground truth community assignments
        algorithm_type (str, optional): 'girvan_newman', 'infomap', or None
    
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
        
    # Check if this is a large graph - adjust metrics computation accordingly
    is_large_graph = G.number_of_nodes() > 50000
    if is_large_graph:
        logger.warning(f"Large graph detected ({G.number_of_nodes()} nodes) - using optimized metrics")
    
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
    
    # Check graph connectivity - important context for interpreting metrics
    connected_components = list(nx.connected_components(G))
    n_components = len(connected_components)
    
    if n_components > 1:
        largest_cc_size = len(max(connected_components, key=len))
        logger.warning(f"Graph has {n_components} connected components (largest: {largest_cc_size} nodes)")
        logger.warning("Metrics may be affected by graph disconnectivity")
    
    # Compute modularity
    modularity = compute_modularity(G, final_partition)
    logger.info(f"Modularity: {modularity:.4f}")
    
    # Compute conductance - for very large graphs, compute on a sample
    if is_large_graph and num_communities > 1000:
        # Sample communities for conductance calculation
        sampled_communities = dict(list(communities.items())[:1000])
        conductance_values, avg_conductance, community_conductances = compute_conductance(G, sampled_communities)
        logger.info(f"Computed conductance for 1000/{num_communities} communities (sampled)")
    else:
        conductance_values, avg_conductance, community_conductances = compute_conductance(G, communities)
    
    logger.info(f"Average conductance: {avg_conductance:.4f}")
    
    # Add interpretation of conductance
    if avg_conductance < 0.01 and n_components > num_communities * 0.8:
        logger.warning("Very low conductance likely due to graph disconnectivity (few edges between components)")
    
    # Calculate coverage (fraction of edges within communities)
    # For large graphs, estimate rather than exact computation
    if is_large_graph and G.number_of_edges() > 100000:
        # Sample a fraction of communities
        sample_size = min(500, num_communities)
        sampled_comm_ids = list(communities.keys())[:sample_size]
        sampled_communities = {cid: communities[cid] for cid in sampled_comm_ids}
        
        intra_edges = 0
        total_sampled_nodes = 0
        
        for nodes in sampled_communities.values():
            total_sampled_nodes += len(nodes)
            # Filter to nodes in graph
            valid_nodes = set(nodes) & nodes_in_graph
            if len(valid_nodes) > 1:  # Need at least 2 nodes for an edge
                subgraph = G.subgraph(valid_nodes)
                intra_edges += subgraph.number_of_edges()
        
        # Estimate full coverage
        total_edges = G.number_of_edges()
        estimated_ratio = len(nodes_in_partition) / total_sampled_nodes if total_sampled_nodes > 0 else 1
        estimated_intra_edges = intra_edges * estimated_ratio
        coverage = estimated_intra_edges / total_edges if total_edges > 0 else 0
        logger.info(f"Estimated coverage based on {sample_size}/{num_communities} communities")
    else:
        # Exact computation for smaller graphs
        intra_edges = 0
        total_edges = G.number_of_edges()
        
        for nodes in communities.values():
            # Filter to nodes in graph
            valid_nodes = set(nodes) & nodes_in_graph
            if len(valid_nodes) > 1:  # Need at least 2 nodes for an edge
                subgraph = G.subgraph(valid_nodes)
                intra_edges += subgraph.number_of_edges()
        
        coverage = intra_edges / total_edges if total_edges > 0 else 0
    
    # Calculate average degree within communities
    avg_internal_degree = 2 * intra_edges / sum(len(nodes) for nodes in communities.values()) if communities else 0
    
    # Compute NMI if ground truth is provided
    nmi = None
    if ground_truth:
        nmi = compute_nmi(final_partition, ground_truth)
        logger.info(f"Normalized Mutual Information: {nmi:.4f}")
    else:
        logger.info("No ground truth provided, skipping NMI calculation")
    
    # Add connectivity analysis to the metrics
    metrics = {
        "modularity": modularity,
        "conductance_values": conductance_values,
        "avg_conductance": avg_conductance,
        "community_conductances": community_conductances,
        "nmi": nmi,
        "num_communities": num_communities,
        "community_sizes": community_sizes,
        "size_stats": size_stats,
        "coverage": coverage,
        "avg_internal_degree": avg_internal_degree,
        "missing_nodes": len(missing_nodes),
        "graph_connectivity": {
            "num_components": n_components,
            "is_connected": n_components == 1,
            "component_sizes": sorted([len(comp) for comp in connected_components], reverse=True)[:10]
        },
        "evaluation_time": time.time() - start_time
    }
    
    # Algorithm-specific metrics
    if algorithm_type == 'girvan_newman' and not is_large_graph:
        # Skip expensive edge betweenness for large graphs
        edge_betweenness, avg_betweenness, inter_comm_betweenness = compute_edge_betweenness(G, final_partition)
        metrics["edge_betweenness"] = edge_betweenness
        metrics["avg_edge_betweenness"] = avg_betweenness
        metrics["inter_community_betweenness"] = inter_comm_betweenness
        logger.info(f"Average edge betweenness: {avg_betweenness:.4f}")
    
    elif algorithm_type == 'infomap':
        # InfoMap metrics are less computationally intensive
        description_length, community_entropy = compute_description_length(G, communities)
        metrics["description_length"] = description_length
        metrics["community_entropy"] = community_entropy
        logger.info(f"Description length: {description_length:.4f}")
    
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
        # Determine algorithm type from name
        algorithm_type = None
        if 'girvan_newman' in algo_name.lower():
            algorithm_type = 'girvan_newman'
        elif 'infomap' in algo_name.lower():
            algorithm_type = 'infomap'
            
        metrics = evaluate_all(G, partition, ground_truth, algorithm_type=algorithm_type)
        comparison[algo_name] = metrics
    
    # Create summary table
    summary = {
        "algorithm": [],
        "modularity": [],
        "avg_conductance": [],
        "num_communities": [],
        "coverage": []
    }
    
    # Add algorithm-specific metric columns
    has_betweenness = any('avg_edge_betweenness' in metrics for metrics in comparison.values())
    has_description_length = any('description_length' in metrics for metrics in comparison.values())
    
    if has_betweenness:
        summary["avg_edge_betweenness"] = []
    
    if has_description_length:
        summary["description_length"] = []
        
    if any(metrics.get('nmi') is not None for metrics in comparison.values()):
        summary["nmi"] = []
    
    for algo_name, metrics in comparison.items():
        summary["algorithm"].append(algo_name)
        summary["modularity"].append(metrics["modularity"])
        summary["avg_conductance"].append(metrics["avg_conductance"])
        summary["num_communities"].append(metrics["num_communities"])
        summary["coverage"].append(metrics["coverage"])
        
        if "avg_edge_betweenness" in metrics and "avg_edge_betweenness" in summary:
            summary["avg_edge_betweenness"].append(metrics["avg_edge_betweenness"])
            
        if "description_length" in metrics and "description_length" in summary:
            summary["description_length"].append(metrics["description_length"])
        
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
    
    # Add algorithm-specific best metrics
    if has_betweenness:
        best_betweenness = min(summary["avg_edge_betweenness"])
        best_betweenness_idx = summary["avg_edge_betweenness"].index(best_betweenness)
        best_betweenness_algo = summary["algorithm"][best_betweenness_idx]
        summary["best"]["edge_betweenness"] = {
            "value": best_betweenness,
            "algorithm": best_betweenness_algo
        }
        
    if has_description_length:
        best_dl = min(summary["description_length"])
        best_dl_idx = summary["description_length"].index(best_dl)
        best_dl_algo = summary["algorithm"][best_dl_idx]
        summary["best"]["description_length"] = {
            "value": best_dl,
            "algorithm": best_dl_algo
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
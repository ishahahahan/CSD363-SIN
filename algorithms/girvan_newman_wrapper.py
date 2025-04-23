import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from collections import defaultdict
import logging
import time

# Change from relative import to absolute import
from evaluation import compute_modularity, track_algorithm_metrics, evaluate_all

logger = logging.getLogger('community_pipeline')

def run_girvan_newman_with_tracking(G, max_communities=None, max_iterations=None):
    """
    Run Girvan-Newman algorithm with tracking of metrics at each step
    
    Args:
        G (networkx.Graph): Input graph
        max_communities (int, optional): Stop when this many communities are found
        max_iterations (int, optional): Maximum number of edge removals
        
    Returns:
        tuple: (final partition dict, dict of tracked metrics, list of intermediate partitions)
    """
    logger.info("Running Girvan-Newman algorithm with metric tracking")
    start_time = time.time()
    
    # For very large graphs, apply optimizations
    if G.number_of_nodes() > 10000:
        logger.warning(f"Large graph detected ({G.number_of_nodes()} nodes). Using optimized GN algorithm.")
        return run_optimized_girvan_newman(G, max_communities, max_iterations)
    
    # Deep copy to avoid modifying the original graph
    graph_copy = G.copy()
    
    # Initialize variables to track progress
    iteration = 0
    current_communities = 1  # Start with 1 (assuming connected graph)
    best_modularity = -1
    best_partition = None
    
    # List to store partitions at each step
    all_partitions = []
    
    # Store edge betweenness values at each step (but limit number stored to save memory)
    edge_betweenness_history = []
    store_history_frequency = max(1, min(5, G.number_of_nodes() // 5000))  # Adjust storage frequency based on graph size
    
    # Set a maximum time limit for the algorithm
    max_time = 600  # 10 minutes
    
    # Run the algorithm
    communities_generator = girvan_newman(graph_copy)
    
    # Convert to list and track metrics for each step
    for communities in communities_generator:
        iteration += 1
        
        # Check if we've exceeded time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            logger.warning(f"Girvan-Newman stopped after {elapsed_time:.2f}s (time limit reached)")
            break
        
        # Convert communities to partition dict
        partition = {}
        for i, community in enumerate(communities):
            for node in community:
                partition[node] = i
        
        # Store current partition
        all_partitions.append(partition)
        
        # Calculate edge betweenness (less frequently for large graphs)
        if iteration % store_history_frequency == 0:
            # Use approximation for large graphs
            if G.number_of_nodes() > 5000:
                # Use approximate betweenness with sampling
                k = min(1000, G.number_of_nodes() // 10)
                edge_betweenness = nx.edge_betweenness_centrality(graph_copy, k=k)
            else:
                edge_betweenness = nx.edge_betweenness_centrality(graph_copy)
            edge_betweenness_history.append(edge_betweenness)
        
        # Calculate modularity
        modularity = compute_modularity(G, partition)
        
        # Track the best partition
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition.copy()
        
        # Count communities
        current_communities = len(set(partition.values()))
        
        # Log progress
        if iteration % 5 == 0:
            logger.info(f"Iteration {iteration}, Communities: {current_communities}, Modularity: {modularity:.4f}")
        
        # Check stopping conditions
        if max_communities is not None and current_communities >= max_communities:
            logger.info(f"Reached target number of communities: {current_communities}")
            break
            
        if max_iterations is not None and iteration >= max_iterations:
            logger.info(f"Reached maximum iterations: {iteration}")
            break
    
    # Use the final partition if no best was found
    if best_partition is None and all_partitions:
        best_partition = all_partitions[-1]
    
    # If still None, create a default partition
    if best_partition is None:
        best_partition = {node: 0 for node in G.nodes()}
    
    # Track all metrics for the algorithm steps
    tracked_metrics = track_algorithm_metrics(
        G, all_partitions, algorithm_type='girvan_newman'
    )
    
    # Add edge betweenness history
    tracked_metrics['edge_betweenness_history'] = edge_betweenness_history
    
    logger.info(f"Girvan-Newman completed in {time.time() - start_time:.2f}s")
    logger.info(f"Final communities: {len(set(best_partition.values()))}")
    logger.info(f"Best modularity: {best_modularity:.4f}")
    
    return best_partition, tracked_metrics, all_partitions

def run_optimized_girvan_newman(G, max_communities=None, max_iterations=None):
    """
    Run an optimized version of Girvan-Newman for large graphs.
    Use sampling and early stopping to improve performance.
    
    Args:
        G (networkx.Graph): Input graph
        max_communities (int, optional): Stop when this many communities are found
        max_iterations (int, optional): Maximum number of edge removals
        
    Returns:
        tuple: (final partition dict, dict of tracked metrics, list of intermediate partitions)
    """
    start_time = time.time()
    logger.info(f"Using optimized Girvan-Newman for large graph ({G.number_of_nodes()} nodes)")
    
    # Initialize tracking variables
    all_partitions = []
    edge_betweenness_history = []
    best_modularity = -1
    best_partition = None
    
    # Limit iterations for large graphs
    if max_iterations is None:
        max_iterations = min(50, G.number_of_nodes() // 2000)
    
    # Start with connected components as initial communities
    components = list(nx.connected_components(G))
    logger.info(f"Starting with {len(components)} connected components")
    
    # If already many components, we can use them directly
    if len(components) > max(20, max_communities or 0):
        logger.info("Graph already has many connected components, using them as communities")
        partition = {}
        for i, component in enumerate(components):
            for node in component:
                partition[node] = i
        
        # We'll skip the expensive edge removal process
        all_partitions.append(partition)
        
        # Calculate modularity
        modularity = compute_modularity(G, partition)
        logger.info(f"Initial modularity from components: {modularity:.4f}")
        
        # Create minimal tracking metrics
        tracked_metrics = {
            'steps': 1,
            'modularity': [modularity],
            'num_communities': [len(components)],
            'conductance': [0.0],  # We'll compute proper conductance later
            'coverage': [1.0]
        }
        
        return partition, tracked_metrics, all_partitions
    
    # If graph is too large, work on the largest connected component
    if G.number_of_nodes() > 20000:
        largest_cc = max(nx.connected_components(G), key=len)
        logger.info(f"Focusing on largest connected component ({len(largest_cc)} nodes)")
        G_sub = G.subgraph(largest_cc).copy()
    else:
        G_sub = G.copy()
    
    # Initial partition: all nodes in one community
    iteration = 0
    current_partition = {node: 0 for node in G.nodes()}
    all_partitions.append(current_partition.copy())
    
    # For each component, apply a limited version of GN
    for comp_idx, component in enumerate(components):
        if len(component) < 10:  # Skip tiny components
            continue
            
        # Extract subgraph for this component
        subgraph = G.subgraph(component).copy()
        
        # Only process larger components, up to a few
        if len(component) >= 100 and comp_idx < 5:
            logger.info(f"Processing component {comp_idx+1} with {len(component)} nodes")
            
            # Calculate edge betweenness (with sampling for large components)
            k = min(1000, len(component) // 2)
            edge_betweenness = nx.edge_betweenness_centrality(subgraph, k=k)
            
            # Sort edges by betweenness
            sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
            
            # Remove top edges to split the component
            max_edges_to_remove = min(20, len(sorted_edges) // 10)
            edges_to_remove = [edge for edge, _ in sorted_edges[:max_edges_to_remove]]
            
            # Remove edges
            subgraph.remove_edges_from(edges_to_remove)
            
            # Get new communities from connected components
            new_components = list(nx.connected_components(subgraph))
            
            if len(new_components) > 1:
                logger.info(f"Split component into {len(new_components)} parts")
                
                # Assign new community IDs
                for i, new_comp in enumerate(new_components):
                    new_comm_id = next_community_id + i if 'next_community_id' in locals() else i
                    for node in new_comp:
                        current_partition[node] = new_comm_id
                
                if 'next_community_id' in locals():
                    next_community_id += len(new_components)
                else:
                    next_community_id = len(new_components)
                
                # Save this partition
                all_partitions.append(current_partition.copy())
                
                # Calculate modularity
                modularity = compute_modularity(G, current_partition)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_partition = current_partition.copy()
                
                # Log progress
                logger.info(f"Step {len(all_partitions)}, Communities: {next_community_id}, Modularity: {modularity:.4f}")
    
    # If we found a good partition, use it
    if best_partition is None and all_partitions:
        best_partition = all_partitions[-1]
    
    # If still None, use initial partition
    if best_partition is None:
        best_partition = {node: 0 for node in G.nodes()}
    
    # Track metrics
    tracked_metrics = track_algorithm_metrics(
        G, all_partitions, algorithm_type='girvan_newman'
    )
    
    # Add empty edge betweenness history to maintain interface
    tracked_metrics['edge_betweenness_history'] = []
    
    logger.info(f"Optimized Girvan-Newman completed in {time.time() - start_time:.2f}s")
    logger.info(f"Final communities: {len(set(best_partition.values()))}")
    if best_modularity > -1:
        logger.info(f"Best modularity: {best_modularity:.4f}")
    
    return best_partition, tracked_metrics, all_partitions

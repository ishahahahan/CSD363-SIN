import networkx as nx
import community as community_louvain
from collections import defaultdict
import logging
import time
import numpy as np
from ..evaluation import compute_modularity, track_algorithm_metrics, evaluate_all, compute_description_length

logger = logging.getLogger('community_pipeline')

# Note: This assumes you have a proper InfoMap implementation available
# If using infomap package: pip install infomap
try:
    import infomap
    HAS_INFOMAP = True
except ImportError:
    HAS_INFOMAP = False
    logger.warning("Infomap package not found. Install with: pip install infomap")

def run_infomap_with_tracking(G, num_trials=10, two_level=True):
    """
    Run InfoMap algorithm with tracking of metrics at each step
    
    Args:
        G (networkx.Graph): Input graph
        num_trials (int): Number of trials to run (InfoMap is stochastic)
        two_level (bool): If True, use two-level map equation
        
    Returns:
        tuple: (final partition dict, dict of tracked metrics, list of intermediate partitions)
    """
    if not HAS_INFOMAP:
        logger.error("Infomap package not installed. Cannot run InfoMap algorithm.")
        # Return empty results
        return {}, {}, []
        
    logger.info("Running InfoMap algorithm with metric tracking")
    start_time = time.time()
    
    # Create Infomap object
    im = infomap.Infomap("--two-level" if two_level else "")
    
    # Add nodes and edges to Infomap network
    for i, n in enumerate(G.nodes()):
        im.add_node(i, n)  # Node index, node name
    
    for e in G.edges():
        im.add_link(G.nodes.index(e[0]), G.nodes.index(e[1]))
    
    # List to store partitions at each step
    all_partitions = []
    description_length_history = []
    
    # Store best partition
    best_partition = None
    best_description_length = float('inf')
    
    # Run multiple trials
    for trial in range(num_trials):
        logger.info(f"Running InfoMap trial {trial+1}/{num_trials}")
        
        # Run the algorithm
        im.run()
        
        # Get the partition
        partition = {}
        for node in im.tree:
            if node.is_leaf:
                module_id = node.module_id
                node_id = node.node_id
                if node_id < len(G):
                    node_name = list(G.nodes())[node_id]
                    partition[node_name] = module_id
        
        # Store current partition
        all_partitions.append(partition)
        
        # Calculate description length
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        dl, _ = compute_description_length(G, communities)
        description_length_history.append(dl)
        
        logger.info(f"Trial {trial+1} - Communities: {len(set(partition.values()))}, "
                   f"Description Length: {dl:.4f}")
        
        # Track the best partition
        if dl < best_description_length:
            best_description_length = dl
            best_partition = partition.copy()
    
    # Use the final partition if no best was found
    if best_partition is None and all_partitions:
        best_partition = all_partitions[-1]
    
    # If still None, create a default partition
    if best_partition is None:
        best_partition = {node: 0 for node in G.nodes()}
    
    # Track all metrics for the algorithm steps
    tracked_metrics = track_algorithm_metrics(
        G, all_partitions, algorithm_type='infomap'
    )
    
    # Add description length history
    tracked_metrics['description_length_history'] = description_length_history
    
    # Final evaluation with algorithm type
    final_metrics = evaluate_all(G, best_partition, algorithm_type='infomap')
    
    logger.info(f"InfoMap completed in {time.time() - start_time:.2f}s")
    logger.info(f"Final communities: {len(set(best_partition.values()))}")
    logger.info(f"Best description length: {best_description_length:.4f}")
    
    return best_partition, tracked_metrics, all_partitions

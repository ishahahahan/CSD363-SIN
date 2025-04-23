import time
import logging
import networkx as nx
import community as community_louvain
import igraph as ig
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)

def run_louvain(G):
    """
    Run the Louvain community detection algorithm on the graph.
    
    Args:
        G (networkx.Graph): Input graph
    
    Returns:
        tuple: (partition dict mapping node->community, communities dict mapping community->[nodes list])
    """
    logger.info("Running Louvain community detection")
    start_time = time.time()
    
    # Run Louvain algorithm
    partition = community_louvain.best_partition(G)
    
    # Create a mapping from community ID to list of nodes
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    # Compute modularity
    modularity = community_louvain.modularity(partition, G)
    
    elapsed = time.time() - start_time
    logger.info(f"Louvain completed in {elapsed:.2f} seconds")
    logger.info(f"Found {len(communities)} communities with modularity {modularity:.4f}")
    
    if elapsed > 60:
        logger.warning(f"Louvain took {elapsed:.2f} seconds, which exceeds the 60-second threshold")
    
    return partition, communities

def refine_girvan_newman(G, communities, size_threshold, target_subcommunities, max_iterations=None):
    """
    Refine large communities using Girvan-Newman algorithm.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary of community_id -> list of nodes
        size_threshold (int): Communities larger than this will be refined
        target_subcommunities (int): Target number of subcommunities after refinement
        max_iterations (int, optional): Maximum number of iterations for Girvan-Newman
        
    Returns:
        dict: Updated partition mapping node -> community_id
    """
    logger = logging.getLogger('community_pipeline')
    logger.info(f"Refining communities larger than {size_threshold} nodes")
    
    # Get the original partition
    partition = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            partition[node] = comm_id
    
    # Find large communities that need refinement
    large_communities = {
        comm_id: nodes for comm_id, nodes in communities.items()
        if len(nodes) > size_threshold
    }
    
    logger.info(f"Found {len(large_communities)} communities larger than threshold")
    
    if not large_communities:
        logger.info("No communities need refinement, returning original partition")
        return partition
    
    # Start with highest community ID to avoid conflicts
    next_community_id = max(communities.keys()) + 1
    
    # Process each large community
    for comm_id, nodes in large_communities.items():
        logger.info(f"Refining community {comm_id} with {len(nodes)} nodes")
        
        # Extract the subgraph for this community
        subgraph = G.subgraph(nodes).copy()
        
        # Skip if subgraph is too small or has no edges
        if subgraph.number_of_nodes() < 3 or subgraph.number_of_edges() < 2:
            logger.info(f"Skipping community {comm_id}: too small for meaningful refinement")
            continue
            
        # Determine target number of subcommunities based on size
        target = min(
            target_subcommunities,
            subgraph.number_of_nodes() // 10
        )
        target = max(2, target)  # At least split into 2
        
        try:
            # Run Girvan-Newman with a limit on iterations or components
            from algorithms.girvan_newman_wrapper import run_girvan_newman_with_tracking
            
            logger.info(f"Running Girvan-Newman on community {comm_id} to find {target} subcommunities")
            
            # Run GN with tracking and parameter passing
            subpartition, metrics, _ = run_girvan_newman_with_tracking(
                subgraph, 
                max_communities=target,
                max_iterations=max_iterations
            )
            
            # Count actual subcommunities found
            subcommunities = set(subpartition.values())
            num_subcommunities = len(subcommunities)
            
            logger.info(f"Found {num_subcommunities} subcommunities for community {comm_id}")
            
            # Only apply refinement if we found multiple communities
            if num_subcommunities > 1:
                # Map the subcommunities to new community IDs
                id_mapping = {old_id: next_community_id + i for i, old_id in enumerate(subcommunities)}
                
                # Update the partition for this community's nodes
                for node, subcomm_id in subpartition.items():
                    partition[node] = id_mapping[subcomm_id]
                
                # Update next_community_id
                next_community_id += num_subcommunities
                
                logger.info(f"Applied refinement: split community {comm_id} into {num_subcommunities} communities")
            else:
                logger.info(f"No meaningful partitioning found for community {comm_id}")
                
        except Exception as e:
            logger.error(f"Error refining community {comm_id}: {str(e)}")
            # Continue with next community
            continue
    
    logger.info(f"Refinement complete. Partition now has {len(set(partition.values()))} communities")
    return partition

def enhance_infomap(G, partition, communities, modularity_threshold):
    """
    Enhance communities with low modularity using Infomap algorithm.
    
    Args:
        G (networkx.Graph): Input graph
        partition (dict): Current partition mapping node->community
        communities (dict): Dictionary mapping community IDs to lists of nodes
        modularity_threshold (float): Threshold below which to apply Infomap
    
    Returns:
        dict: Enhanced partition mapping node->community
    """
    try:
        from infomap import Infomap
    except ImportError:
        logger.warning("Infomap package not available. Skipping enhancement step.")
        return partition
    
    logger.info(f"Enhancing communities with modularity < {modularity_threshold}")
    start_time = time.time()
    
    # Make a copy of the partition to update
    enhanced_partition = partition.copy()
    next_community_id = max(communities.keys()) + 1 if communities else 0
    
    # Check each community's local modularity
    low_modularity_communities = []
    
    for comm_id, nodes in tqdm(communities.items(), desc="Evaluating community modularity"):
        if len(nodes) < 10:  # Skip very small communities
            continue
            
        G_sub = G.subgraph(nodes).copy()
        
        if G_sub.number_of_edges() == 0:  # Skip communities with no edges
            continue
            
        # Create a trivial partition where all nodes are in the same community
        sub_partition = {node: 0 for node in G_sub.nodes()}
        
        # Calculate local modularity
        local_modularity = community_louvain.modularity(sub_partition, G_sub)
        
        if local_modularity < modularity_threshold:
            logger.debug(f"Community {comm_id} has low modularity ({local_modularity:.4f}), will enhance")
            low_modularity_communities.append((comm_id, nodes))
    
    logger.info(f"Found {len(low_modularity_communities)} communities with low modularity")
    
    # Process each low-modularity community with Infomap
    for comm_id, nodes in tqdm(low_modularity_communities, desc="Enhancing with Infomap"):
        G_sub = G.subgraph(nodes).copy()
        
        # Convert to igraph for Infomap
        edges = list(G_sub.edges())
        ig_graph = ig.Graph.TupleList(edges, directed=False)
        
        # Map between networkx and igraph vertex indices
        nodes_list = list(G_sub.nodes())
        nx_to_ig = {node: i for i, node in enumerate(nodes_list)}
        ig_to_nx = {i: nodes_list[i] for i in range(len(nodes_list))}
        
        # Run Infomap
        infomap = Infomap("--two-level")
        
        # Add edges to Infomap network
        for edge in edges:
            source, target = edge
            infomap.add_link(nx_to_ig[source], nx_to_ig[target])
        
        # Run Infomap
        infomap.run()
        
        # Extract and map back the communities
        for node_id, module_id in infomap.get_modules().items():
            if node_id in ig_to_nx:
                nx_node = ig_to_nx[node_id]
                # Assign new community ID
                enhanced_partition[nx_node] = next_community_id + module_id
        
        next_community_id += infomap.num_top_modules
    
    elapsed = time.time() - start_time
    logger.info(f"Infomap enhancement completed in {elapsed:.2f} seconds")
    
    if elapsed > 60:
        logger.warning(f"Infomap enhancement took {elapsed:.2f} seconds, which exceeds the 60-second threshold")
    
    # Update communities dict
    enhanced_communities = defaultdict(list)
    for node, comm_id in enhanced_partition.items():
        enhanced_communities[comm_id].append(node)
    
    logger.info(f"After enhancement: {len(enhanced_communities)} communities")
    
    return enhanced_partition

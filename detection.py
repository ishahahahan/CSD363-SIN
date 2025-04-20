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

def refine_girvan_newman(G, communities, size_threshold, target_subcommunities):
    """
    Refine large communities using Girvan-Newman edge betweenness algorithm.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to lists of nodes
        size_threshold (int): Size threshold for community refinement
        target_subcommunities (int): Target number of subcommunities to find
    
    Returns:
        dict: Updated partition mapping node->community
    """
    logger.info(f"Refining communities larger than {size_threshold} nodes")
    start_time = time.time()
    
    # Start with the original partition
    partition = {node: comm_id for comm_id, nodes in communities.items() for node in nodes}
    next_community_id = max(communities.keys()) + 1 if communities else 0
    
    # Process large communities
    large_communities = {comm_id: nodes for comm_id, nodes in communities.items() 
                        if len(nodes) > size_threshold}
    
    if not large_communities:
        logger.info("No communities exceed the size threshold")
        return partition
    
    logger.info(f"Found {len(large_communities)} communities to refine")
    
    for comm_id, nodes in tqdm(large_communities.items(), desc="Refining large communities"):
        logger.debug(f"Processing community {comm_id} with {len(nodes)} nodes")
        
        # Extract subgraph for this community
        G_sub = G.subgraph(nodes).copy()
        
        # Skip if subgraph is too small or has no edges
        if G_sub.number_of_nodes() < 10 or G_sub.number_of_edges() == 0:
            logger.debug(f"Skipping community {comm_id}: too small or no edges")
            continue
        
        # Compute edge betweenness centrality (approximate for large graphs)
        logger.debug(f"Computing edge betweenness for community {comm_id}")
        sub_start_time = time.time()
        
        if G_sub.number_of_nodes() > 5000:
            logger.debug("Using approximate edge betweenness (k=1000)")
            edge_betweenness = nx.edge_betweenness_centrality(G_sub, k=1000)
        else:
            edge_betweenness = nx.edge_betweenness_centrality(G_sub)
        
        logger.debug(f"Edge betweenness computed in {time.time() - sub_start_time:.2f} seconds")
        
        # Sort edges by betweenness
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
        
        # Iteratively remove edges and check connected components
        edges_to_remove = []
        G_tmp = G_sub.copy()
        current_components = list(nx.connected_components(G_tmp))
        
        logger.debug(f"Starting edge removal with {len(current_components)} component(s)")
        
        for i, ((u, v), _) in enumerate(sorted_edges):
            edges_to_remove.append((u, v))
            
            # Check components every 10 edge removals or at the end
            if (i + 1) % 10 == 0 or i == len(sorted_edges) - 1:
                G_tmp.remove_edges_from(edges_to_remove)
                edges_to_remove = []
                current_components = list(nx.connected_components(G_tmp))
                
                if len(current_components) >= target_subcommunities:
                    logger.debug(f"Reached target of {len(current_components)} subcommunities after removing {i+1} edges")
                    break
        
        # Assign new community IDs to components
        for component in current_components:
            # Skip tiny components (likely noise)
            if len(component) < 5:
                continue
                
            for node in component:
                partition[node] = next_community_id
            next_community_id += 1
    
    elapsed = time.time() - start_time
    logger.info(f"Girvan-Newman refinement completed in {elapsed:.2f} seconds")
    
    if elapsed > 60:
        logger.warning(f"Girvan-Newman refinement took {elapsed:.2f} seconds, which exceeds the 60-second threshold")
    
    # Regenerate communities dict
    new_communities = defaultdict(list)
    for node, comm_id in partition.items():
        new_communities[comm_id].append(node)
    
    logger.info(f"After refinement: {len(new_communities)} communities")
    
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
        nx_to_ig = {node: i for i, node in enumerate(G_sub.nodes())}
        ig_to_nx = {i: node for node, i in nx_to_ig.items()}
        
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

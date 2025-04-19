import time
import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

def plot_communities(G, communities, max_nodes=1000):
    """
    Visualize communities in the graph.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to lists of nodes
        max_nodes (int): Maximum number of nodes to include in visualization
    
    Returns:
        str: Path to saved visualization image
    """
    logger.info("Generating community visualization")
    start_time = time.time()
    
    # Sample graph if too large
    if G.number_of_nodes() > max_nodes:
        logger.info(f"Graph too large ({G.number_of_nodes()} nodes), sampling {max_nodes} nodes")
        
        # Strategy: Sample from each community proportionally to its size
        sampled_nodes = []
        
        # Calculate number of nodes to sample from each community
        total_nodes = sum(len(nodes) for nodes in communities.values())
        comm_sizes = {comm_id: len(nodes) for comm_id, nodes in communities.items()}
        
        # Only include communities with at least 5 nodes
        valid_comms = {cid: size for cid, size in comm_sizes.items() if size >= 5}
        total_valid = sum(valid_comms.values())
        
        # Calculate sampling proportions
        for comm_id, size in valid_comms.items():
            # Ensure each community has at least 5 nodes in the sample
            sample_size = max(5, int((size / total_valid) * max_nodes))
            
            # Sample nodes from this community
            comm_nodes = communities[comm_id]
            if len(comm_nodes) <= sample_size:
                comm_sample = comm_nodes
            else:
                comm_sample = random.sample(comm_nodes, sample_size)
                
            sampled_nodes.extend(comm_sample)
            
        # Cap at max_nodes if we exceeded
        if len(sampled_nodes) > max_nodes:
            sampled_nodes = random.sample(sampled_nodes, max_nodes)
            
        # Create subgraph
        G_viz = G.subgraph(sampled_nodes).copy()
        
        # Update communities to only include sampled nodes
        viz_communities = defaultdict(list)
        for comm_id, nodes in communities.items():
            viz_nodes = [node for node in nodes if node in G_viz]
            if viz_nodes:
                viz_communities[comm_id] = viz_nodes
    else:
        logger.info(f"Using full graph for visualization ({G.number_of_nodes()} nodes)")
        G_viz = G
        viz_communities = communities
    
    # Remove isolated nodes for better visualization
    G_viz.remove_nodes_from(list(nx.isolates(G_viz)))
    
    if len(G_viz) == 0:
        logger.warning("No nodes to visualize after preprocessing")
        return None
    
    logger.info(f"Visualization graph: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")
    
    # Generate a color map
    num_communities = len(viz_communities)
    color_map = plt.cm.get_cmap('tab20', num_communities)
    
    # Create mapping from node to color based on community
    node_colors = []
    node_to_comm = {}
    
    for i, (comm_id, nodes) in enumerate(viz_communities.items()):
        for node in nodes:
            node_to_comm[node] = comm_id
    
    for node in G_viz.nodes():
        if node in node_to_comm:
            # Normalize community ID to [0, 1] range for color mapping
            color_idx = list(viz_communities.keys()).index(node_to_comm[node]) / max(1, num_communities-1)
            node_colors.append(color_map(color_idx))
        else:
            # Default color for nodes without community
            node_colors.append('lightgrey')
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for visualization
    logger.info("Computing layout (this may take a moment)...")
    layout = nx.spring_layout(G_viz, seed=42)
    
    # Draw the network
    nx.draw_networkx(
        G_viz,
        pos=layout,
        node_color=node_colors,
        node_size=50,
        with_labels=False,
        edge_color='lightgrey',
        alpha=0.8
    )
    
    # Label representative nodes (nodes with highest degree in each community)
    labels = {}
    for comm_id, nodes in viz_communities.items():
        # Find the node with highest degree in this community
        if not nodes:
            continue
            
        nodes_in_graph = [n for n in nodes if n in G_viz]
        if not nodes_in_graph:
            continue
            
        representative = max(nodes_in_graph, key=lambda n: G_viz.degree(n))
        labels[representative] = str(comm_id)
    
    nx.draw_networkx_labels(
        G_viz,
        pos=layout,
        labels=labels,
        font_size=12,
        font_weight='bold',
        font_color='black'
    )
    
    # Add plot title and legend
    plt.title(f"Community Structure (showing {G_viz.number_of_nodes()} nodes in {len(viz_communities)} communities)")
    
    # Save the figure
    output_path = "community_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Visualization completed in {elapsed:.2f} seconds and saved to {output_path}")
    
    if elapsed > 60:
        logger.warning(f"Visualization took {elapsed:.2f} seconds, which exceeds the 60-second threshold")
    
    return output_path

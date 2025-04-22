def plot_communities(G, communities):
    """
    Create improved visualizations of community structure with multiple views.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to lists of nodes
    
    Returns:
        str: Path to main visualization file
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import time
    import numpy as np
    import os
    from collections import Counter
    
    start_time = time.time()
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_dir = "community_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- PREPARATION ---
    
    # Remove isolated nodes
    G_viz = G.copy()
    G_viz.remove_nodes_from(list(nx.isolates(G_viz)))
    
    if not G_viz.nodes:
        logger.warning("No nodes to visualize after removing isolates.")
        return None
    
    # Map nodes to communities
    node_to_comm = {}
    for cid, nodes in communities.items():
        for node in nodes:
            if node in G_viz:
                node_to_comm[node] = cid
    
    # Get community statistics
    community_sizes = {cid: len(nodes) for cid, nodes in communities.items()}
    total_communities = len(communities)
    
    # Sort communities by size (largest first)
    largest_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Create a color mapping for communities
    unique_communities = sorted(set(node_to_comm.values()))
    comm_to_color = {cid: i for i, cid in enumerate(unique_communities)}
    
    # Get a suitable colormap with enough distinct colors
    # Using tab20 (20 colors) and cycling if more are needed
    color_map = plt.cm.get_cmap("tab20")
    
    logger.info(f"Preparing to visualize {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges in {total_communities} communities")
    
    # --- VISUALIZATION 1: TOP COMMUNITIES SUMMARY ---
    
    # Only show the top N largest communities
    top_n = min(20, len(largest_communities))
    top_communities = dict(largest_communities[:top_n])
    
    plt.figure(figsize=(12, 8))
    
    # Create bar chart of community sizes
    plt.bar(
        range(len(top_communities)), 
        [size for _, size in largest_communities[:top_n]],
        color=[color_map(i % 20) for i in range(top_n)]
    )
    
    # Add labels
    plt.xlabel('Community ID')
    plt.ylabel('Number of Nodes')
    plt.title(f'Top {top_n} Largest Communities')
    plt.xticks(range(len(top_communities)), [f"C{cid}" for cid, _ in largest_communities[:top_n]], rotation=45)
    
    # Save figure
    summary_path = os.path.join(output_dir, "community_size_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()
    
    logger.info(f"Community size summary saved to {summary_path}")
    
    # --- VISUALIZATION 2: TOP COMMUNITIES SUBGRAPH ---
    
    # Create a subgraph of only the top N communities (more manageable)
    top_community_ids = [cid for cid, _ in largest_communities[:top_n]]
    nodes_in_top = []
    for cid in top_community_ids:
        nodes_in_top.extend([n for n in communities[cid] if n in G_viz])
    
    # Create subgraph
    G_top = G_viz.subgraph(nodes_in_top).copy()
    
    # Compute a layout that preserves community structure
    logger.info(f"Computing layout for top {top_n} communities subgraph...")
    
    # Use a combination of layouts for better results without FA2
    
    # First, position each community separately in a circular layout
    positions = {}
    offset_x, offset_y = 0, 0
    max_radius = 0
    
    for cid in top_community_ids:
        # Get nodes in this community
        comm_nodes = [n for n in communities[cid] if n in G_top]
        if len(comm_nodes) < 2:
            continue
            
        # Create a subgraph for this community
        G_comm = G_top.subgraph(comm_nodes).copy()
        
        # Use circular layout for this community
        radius = np.sqrt(len(comm_nodes)) * 2  # Scale radius by sqrt of community size
        comm_pos = nx.circular_layout(G_comm, scale=radius)
        
        # Add offset to separate communities
        for node, pos in comm_pos.items():
            positions[node] = pos + np.array([offset_x, offset_y])
        
        # Update offset for next community (arrange in a grid)
        offset_x += radius * 4
        max_radius = max(max_radius, radius)
        if offset_x > 40:  # Start a new row after certain width
            offset_x = 0
            offset_y += max_radius * 4
            max_radius = 0
    
    # For any remaining nodes not positioned, use random layout
    remaining_nodes = [n for n in G_top.nodes() if n not in positions]
    if remaining_nodes:
        remaining_pos = nx.random_layout(G_top.subgraph(remaining_nodes))
        # Place these at the bottom
        for node, pos in remaining_pos.items():
            positions[node] = pos + np.array([20, offset_y + max_radius * 4])
    
    # Now refine the layout with a few iterations of spring_layout, starting from our initial positions
    try:
        # Use spring layout with our initial positions as a starting point
        # This will pull connected nodes from different communities closer together
        positions = nx.spring_layout(
            G_top, 
            pos=positions,
            iterations=50,  # Limit iterations for speed
            k=0.5,         # Optimal distance between nodes
            seed=42
        )
    except:
        # If spring layout fails, just use our initial positions
        logger.warning("Spring layout refinement failed, using initial positions")
    
    plt.figure(figsize=(16, 16))
    
    # Get colors for each node based on community
    node_colors = []
    for node in G_top.nodes():
        comm_id = node_to_comm.get(node)
        color_idx = comm_to_color.get(comm_id, 0) % 20  # Cycle through colors if more than 20
        node_colors.append(color_map(color_idx))
    
    # Calculate node size based on degree (more connected = larger)
    node_degrees = dict(G_top.degree())
    max_degree = max(node_degrees.values()) if node_degrees else 1
    node_sizes = [20 + 100 * (node_degrees[n] / max_degree) for n in G_top.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(
        G_top, 
        positions, 
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )
    
    # Draw edges with transparency to reduce clutter
    nx.draw_networkx_edges(
        G_top, 
        positions, 
        alpha=0.2, 
        width=0.3, 
        edge_color='gray'
    )
    
    # Add labels for communities
    community_centers = {}
    for cid in top_community_ids:
        nodes_in_comm = [n for n in communities[cid] if n in G_top]
        if nodes_in_comm:
            # Calculate the center of this community
            x_coords = [positions[n][0] for n in nodes_in_comm]
            y_coords = [positions[n][1] for n in nodes_in_comm]
            center_x = sum(x_coords) / len(nodes_in_comm)
            center_y = sum(y_coords) / len(nodes_in_comm)
            community_centers[cid] = (center_x, center_y)
            
            # Add community label
            plt.text(
                center_x, center_y, 
                f"C{cid}\n({community_sizes[cid]})", 
                fontsize=14, 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                horizontalalignment='center'
            )
    
    # Add legend with colors
    handles = []
    labels = []
    for i, (cid, size) in enumerate(largest_communities[:min(10, top_n)]):  # Show only top 10 in legend to avoid clutter
        color_idx = comm_to_color.get(cid, 0) % 20
        patch = plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w',
            markerfacecolor=color_map(color_idx),
            markersize=10, 
            label=f'C{cid} ({size} nodes)'
        )
        handles.append(patch)
        labels.append(f'C{cid} ({size} nodes)')
    
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.axis('off')  # Hide axes
    plt.title(f"Top {top_n} Communities Structure ({G_top.number_of_nodes()} nodes, {G_top.number_of_edges()} edges)")
    
    # Save figure
    top_communities_path = os.path.join(output_dir, "top_communities_visualization.png")
    plt.tight_layout()
    plt.savefig(top_communities_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Top communities visualization saved to {top_communities_path}")
    
    # --- VISUALIZATION 3: INDIVIDUAL COMMUNITY DETAILED VIEWS ---
    
    # Generate detailed visualizations for the top 5 communities
    for idx, (cid, size) in enumerate(largest_communities[:5]):
        # Skip small communities
        if size < 10:
            continue
            
        logger.info(f"Generating detailed view for community {cid} ({size} nodes)")
        
        # Extract the subgraph for this community
        comm_nodes = [n for n in communities[cid] if n in G_viz]
        G_comm = G_viz.subgraph(comm_nodes).copy()
        
        if G_comm.number_of_nodes() < 5:
            continue
        
        # Use spring layout for better visualization of internal structure
        comm_layout = nx.spring_layout(G_comm, seed=42, iterations=100)
        
        plt.figure(figsize=(12, 12))
        
        # Calculate node betweenness centrality to size nodes
        try:
            # Use approximate betweenness for large communities
            if G_comm.number_of_nodes() > 1000:
                bc = nx.betweenness_centrality(G_comm, k=100)
            else:
                bc = nx.betweenness_centrality(G_comm)
                
            # Scale node sizes based on betweenness
            max_bc = max(bc.values()) if bc else 1
            node_sizes = [50 + 200 * (bc[n] / max_bc) for n in G_comm.nodes()]
        except:
            # Fall back to degree if betweenness fails
            node_degrees = dict(G_comm.degree())
            max_degree = max(node_degrees.values()) if node_degrees else 1
            node_sizes = [30 + 100 * (node_degrees[n] / max_degree) for n in G_comm.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_comm,
            comm_layout,
            node_size=node_sizes,
            node_color=[color_map(comm_to_color[cid] % 20)] * G_comm.number_of_nodes(),
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G_comm,
            comm_layout,
            alpha=0.3,
            width=0.5
        )
        
        # Label highest betweenness/degree nodes
        if len(node_sizes) > 0:
            # Get top 5 nodes by size
            node_list = list(G_comm.nodes())
            top_indices = sorted(range(len(node_sizes)), key=lambda i: node_sizes[i], reverse=True)[:5]
            top_nodes = [node_list[i] for i in top_indices]
            
            node_labels = {node: str(node) for node in top_nodes}
            nx.draw_networkx_labels(
                G_comm,
                comm_layout,
                labels=node_labels,
                font_size=8,
                font_weight='bold'
            )
        
        plt.title(f"Detailed View of Community {cid} ({size} nodes, {G_comm.number_of_edges()} edges)")
        plt.axis('off')
        
        # Save figure
        comm_path = os.path.join(output_dir, f"community_{cid}_detailed.png")
        plt.tight_layout()
        plt.savefig(comm_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Detailed view for community {cid} saved to {comm_path}")
    
    # --- VISUALIZATION 4: COMMUNITY INTERCONNECTION GRAPH ---
    
    # Create a graph showing how communities are connected to each other
    comm_graph = nx.Graph()
    
    # Add nodes for each community
    for cid, size in largest_communities[:30]:  # Only use top 30 communities
        comm_graph.add_node(cid, size=size)
    
    # Count edges between communities
    comm_connections = Counter()
    
    # Get edges between different communities
    for u, v in G_viz.edges():
        if u in node_to_comm and v in node_to_comm:
            cu = node_to_comm[u]
            cv = node_to_comm[v]
            if cu != cv and cu in comm_graph.nodes() and cv in comm_graph.nodes():
                if cu < cv:
                    comm_connections[(cu, cv)] += 1
                else:
                    comm_connections[(cv, cu)] += 1
    
    # Add edges to the community graph
    for (u, v), weight in comm_connections.items():
        comm_graph.add_edge(u, v, weight=weight)
    
    # Draw the community graph
    plt.figure(figsize=(14, 14))
    
    # Calculate node sizes based on community size
    max_size = max(attr['size'] for _, attr in comm_graph.nodes(data=True))
    node_sizes = [300 * (attr['size'] / max_size) for _, attr in comm_graph.nodes(data=True)]
    
    # Get edge weights
    edge_weights = [comm_graph[u][v]['weight'] for u, v in comm_graph.edges()]
    
    # Calculate edge widths based on weights
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 * (w / max_weight) for w in edge_weights]
    
    # Draw the network
    pos = nx.spring_layout(comm_graph, k=1, iterations=50)
    nx.draw_networkx_nodes(comm_graph, pos, 
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.7)
    nx.draw_networkx_edges(comm_graph, pos,
                          width=edge_widths,
                          alpha=0.5,
                          edge_color='gray')
    nx.draw_networkx_labels(comm_graph, pos,
                           font_size=8,
                           font_weight='bold')
    
    plt.title("Community Interconnection Graph\n(Node size represents community size, edge width represents number of connections)")
    plt.axis('off')
    
    # Save figure
    comm_graph_path = os.path.join(output_dir, "community_interconnection_graph.png")
    plt.tight_layout()
    plt.savefig(comm_graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Community interconnection graph saved to {comm_graph_path}")
    
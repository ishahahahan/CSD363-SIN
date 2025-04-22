import time
import logging
import time
import logging
import random
import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# def plot_communities(G, communities):
#     import matplotlib.pyplot as plt
#     import networkx as nx
#     from collections import defaultdict
#     import random
#     import time

#     start_time = time.time()

#     G_viz = G.copy()
#     G_viz.remove_nodes_from(list(nx.isolates(G_viz)))

#     if not G_viz.nodes:
#         print("No nodes to visualize.")
#         return None

#     # Step 1: Map each node to its community
#     node_to_comm = {}
#     for i, (cid, nodes) in enumerate(communities.items()):
#         for node in nodes:
#             if node in G_viz:
#                 node_to_comm[node] = i

#     # Step 2: Color nodes by community
#     comm_keys = list(set(node_to_comm.values()))
#     color_map = plt.cm.get_cmap("tab20", len(comm_keys))
#     node_colors = [color_map(node_to_comm.get(node, -1)) for node in G_viz.nodes()]

#     # Step 3: Layout and drawing
#     layout = nx.spring_layout(G_viz, seed=42, k=None)

#     plt.figure(figsize=(16, 14))
#     nx.draw_networkx(
#         G_viz,
#         pos=layout,
#         node_color=node_colors,
#         node_size=40,
#         edge_color="lightgrey",
#         alpha=0.8,
#         with_labels=False,
#     )

#     # Step 4: Label one representative node per community
#     labels = {}
#     for cid, nodes in communities.items():
#         valid_nodes = [n for n in nodes if n in G_viz]
#         if valid_nodes:
#             rep = max(valid_nodes, key=lambda n: G_viz.degree(n))
#             labels[rep] = str(cid)

#     nx.draw_networkx_labels(G_viz, pos=layout, labels=labels, font_size=8, font_color="black")

#     plt.title(f"Community Structure (showing {G_viz.number_of_nodes()} nodes in {len(communities)} communities)")
#     output_path = "full_community_visualization.png"
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"Full community visualization saved to {output_path} in {time.time() - start_time:.2f}s")
#     return output_path

# def plot_communities(G, communities):
#     import matplotlib.pyplot as plt
#     import networkx as nx
#     import time
#     import numpy as np

#     start_time = time.time()

#     G_viz = G.copy()
#     G_viz.remove_nodes_from(list(nx.isolates(G_viz)))

#     if not G_viz.nodes:
#         print("No nodes to visualize.")
#         return None

#     # Map each node to its community
#     node_to_comm = {}
#     for cid, nodes in communities.items():
#         for node in nodes:
#             if node in G_viz:
#                 node_to_comm[node] = cid

#     # Get unique community IDs
#     unique_communities = sorted(set(node_to_comm.values()))
    
#     # Create a mapping from community ID to color index
#     comm_to_color = {cid: i for i, cid in enumerate(unique_communities)}
    
#     # Create color map with enough colors for all communities
#     color_map = plt.cm.get_cmap("tab20", len(unique_communities))
    
#     # Assign colors to nodes based on their community
#     node_colors = [color_map(comm_to_color[node_to_comm.get(node, unique_communities[0])]) 
#                   for node in G_viz.nodes()]

#     # Use a more scalable layout algorithm
#     print("Computing layout (this may take a long time for large graphs)...")
    
#     # Option 1: Random layout (very fast but less informative)
#     layout = nx.random_layout(G_viz, seed=42)
    
#     print("Drawing graph...")
#     plt.figure(figsize=(20, 20))
    
#     # Draw with reduced visual elements for better performance
#     nx.draw(
#         G_viz,
#         pos=layout,
#         node_color=node_colors,
#         node_size=10,  # Smaller nodes
#         edge_color="lightgrey",
#         width=0.1,     # Thinner edges
#         alpha=0.6,
#         with_labels=False,
#     )

#     # Instead of labeling individual nodes or all communities, 
#     # only label the largest communities (top 5-10)
#     community_sizes = {cid: len(nodes) for cid, nodes in communities.items()}
#     largest_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:7]  # Only top 7
    
#     # Label only the largest communities
#     comm_centers = {}
#     for comm_id, _ in largest_communities:
#         valid_nodes = [n for n in communities[comm_id] if n in G_viz and n in layout]
#         if valid_nodes:
#             # Find the center of this community
#             x_coords = [layout[n][0] for n in valid_nodes]
#             y_coords = [layout[n][1] for n in valid_nodes]
#             center_x = sum(x_coords) / len(valid_nodes)
#             center_y = sum(y_coords) / len(valid_nodes)
#             comm_centers[comm_id] = (center_x, center_y)
    
#     # Add community labels with size information
#     for comm_id, center in comm_centers.items():
#         size = community_sizes[comm_id]
#         plt.text(center[0], center[1], f"C{comm_id}\n({size})", 
#                  fontsize=12, fontweight='bold', 
#                  bbox=dict(facecolor='white', alpha=0.7),
#                  horizontalalignment='center')

#     # Add a legend for community colors
#     handles = []
#     labels = []
#     for comm_id, size in largest_communities:
#         patch = plt.Line2D([0], [0], marker='o', color='w', 
#                           markerfacecolor=color_map(comm_to_color[comm_id]), 
#                           markersize=10, label=f'Community {comm_id} ({size} nodes)')
#         handles.append(patch)
#         labels.append(f'Community {comm_id} ({size} nodes)')
    
#     # Place legend outside the main plot
#     plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

#     plt.title(f"Community Structure ({G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges, {len(communities)} communities)")
#     output_path = "full_community_visualization.png"
    
#     print(f"Saving visualization (this may take a while)...")
#     plt.savefig(output_path, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"Full community visualization saved to {output_path} in {time.time() - start_time:.2f}s")
#     return output_path

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
    
    # Use ForceAtlas2 layout (faster and better for community structure)
    try:
        from fa2 import ForceAtlas2
        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            # Performance
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,
            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            # Log
            verbose=False
        )
        
        # Prepare positions dict for FA2
        positions = {node: [np.random.random(), np.random.random()] for node in G_top.nodes()}
        
        # Run ForceAtlas2 layout for a limited number of iterations
        positions = forceatlas2.forceatlas2_networkx_layout(G_top, pos=positions, iterations=100)
        
    except ImportError:
        # Fall back to spring layout if FA2 is not available
        logger.warning("FA2 not available, falling back to spring layout")
        positions = nx.spring_layout(G_top, seed=42, iterations=100)
    
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
    max_weight = max(edge_weights) if edge_weights else 1
    
    # Use a force-directed layout
    pos = nx.spring_layout(comm_graph, k=0.5, iterations=100, seed=42)
    
    # Draw nodes colored by community ID
    nx.draw_networkx_nodes(
        comm_graph, 
        pos,
        node_size=node_sizes,
        node_color=[color_map(comm_to_color[cid] % 20) for cid in comm_graph.nodes()],
        alpha=0.8
    )
    
    # Draw edges with width proportional to weight
    edge_widths = [1 + 5 * (w / max_weight) for w in edge_weights]
    nx.draw_networkx_edges(
        comm_graph,
        pos,
        width=edge_widths,
        alpha=0.5,
        edge_color='gray'
    )
    
    # Add labels
    labels = {cid: f"C{cid}" for cid in comm_graph.nodes()}
    nx.draw_networkx_labels(
        comm_graph,
        pos,
        labels=labels,
        font_size=10,
        font_weight='bold'
    )
    
    plt.title("Community Interconnection Graph (Top 30 Communities)")
    plt.axis('off')
    
    # Save figure
    interconnect_path = os.path.join(output_dir, "community_interconnections.png")
    plt.tight_layout()
    plt.savefig(interconnect_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Community interconnection graph saved to {interconnect_path}")
    
    # --- FINAL OUTPUT ---
    
    elapsed = time.time() - start_time
    logger.info(f"All visualizations completed in {elapsed:.2f} seconds")
    
    # Create an HTML report with all visualizations
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Community Detection Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            h1, h2 {{ color: #333; }}
            .section {{ margin-bottom: 40px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Community Detection Analysis Results</h1>
        
        <div class="stats">
            <h2>Summary Statistics</h2>
            <p>Total nodes: {G.number_of_nodes()}</p>
            <p>Total edges: {G.number_of_edges()}</p>
            <p>Total communities: {len(communities)}</p>
            <p>Largest community: {largest_communities[0][0]} ({largest_communities[0][1]} nodes)</p>
        </div>
        
        <div class="section">
            <h2>Community Size Distribution</h2>
            <img src="community_size_summary.png" alt="Community Size Summary">
        </div>
        
        <div class="section">
            <h2>Top Communities Overview</h2>
            <img src="top_communities_visualization.png" alt="Top Communities Visualization">
        </div>
        
        <div class="section">
            <h2>Community Interconnections</h2>
            <img src="community_interconnections.png" alt="Community Interconnections">
        </div>
        
        <div class="section">
            <h2>Detailed Community Views</h2>
    """
    
    # Add detailed community views
    for idx, (cid, _) in enumerate(largest_communities[:5]):
        html_content += f"""
            <h3>Community {cid}</h3>
            <img src="community_{cid}_detailed.png" alt="Community {cid} Detailed View">
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    html_path = os.path.join(output_dir, "community_report.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Comprehensive visualization report saved to {html_path}")
    
    return html_path

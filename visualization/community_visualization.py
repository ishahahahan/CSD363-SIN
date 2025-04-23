import os
import time
import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

logger = logging.getLogger('community_pipeline')

def plot_communities(G, communities):
    """
    Create visualizations of community structure with multiple views.
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to lists of nodes
    
    Returns:
        str: Path to main visualization file
    """
    start_time = time.time()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = "community_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze graph structure for context
        n_components = nx.number_connected_components(G)
        connected_components = list(nx.connected_components(G))
        largest_cc_size = len(max(connected_components, key=len)) if connected_components else 0
        
        # Remove isolated nodes
        G_viz = G.copy()
        isolated_before = len(list(nx.isolates(G_viz)))
        G_viz.remove_nodes_from(list(nx.isolates(G_viz)))
        isolated_removed = isolated_before - len(list(nx.isolates(G_viz)))
        
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
        color_map = plt.cm.get_cmap("tab20")
        
        logger.info(f"Preparing to visualize {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges in {total_communities} communities")
        if isolated_removed > 0:
            logger.info(f"Removed {isolated_removed} isolated nodes for better visualization")
        
        # --- VISUALIZATION 1: COMMUNITY SIZE SUMMARY ---
        
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
        
        try:
            # Use spring layout as a fallback
            positions = nx.spring_layout(G_top, seed=42, iterations=100)
        except:
            # If even spring layout fails, use random layout
            positions = nx.random_layout(G_top, seed=42)
        
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
        
        # Add community labels
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
        
        # --- VISUALIZATION 3: COMMUNITY INTERCONNECTIONS ---
        logger.info("Creating community interconnection visualization...")
        
        # Create community graph where nodes are communities and edges represent connections
        community_graph = nx.Graph()
        
        # Add nodes (each community)
        for comm_id, size in largest_communities[:30]:  # Limit to top 30 communities
            # Only add communities that have nodes in the visualization graph
            if any(node in G_viz for node in communities[comm_id]):
                community_graph.add_node(comm_id, size=size)
        
        # Add edges between communities
        inter_community_edges = defaultdict(int)
        
        # Count edges between communities
        for u, v in G_viz.edges():
            if u in node_to_comm and v in node_to_comm:
                comm_u = node_to_comm[u]
                comm_v = node_to_comm[v]
                if comm_u != comm_v:
                    edge = tuple(sorted([comm_u, comm_v]))  # Ensure consistent ordering
                    inter_community_edges[edge] += 1
        
        # Add edges to community graph with weights
        for (comm_u, comm_v), weight in inter_community_edges.items():
            if comm_u in community_graph and comm_v in community_graph:
                community_graph.add_edge(comm_u, comm_v, weight=weight)
        
        # Visualize community interconnections
        plt.figure(figsize=(16, 12))
        
        # Node sizes based on community size
        node_sizes = [community_graph.nodes[comm_id]['size'] * 0.5 for comm_id in community_graph]
        node_sizes = [max(100, min(2000, size)) for size in node_sizes]  # Constrain sizes
        
        # Node colors based on community ID
        node_colors = [color_map(comm_to_color.get(comm_id, 0) % 20) for comm_id in community_graph]
        
        # Get edge weights for line thickness
        edge_weights = [community_graph[u][v]['weight'] for u, v in community_graph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 3 * (w / max_weight) for w in edge_weights]
        
        # Compute layout
        try:
            pos = nx.spring_layout(community_graph, k=2.0, seed=42)
        except:
            pos = nx.random_layout(community_graph)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            community_graph, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw edges with width based on weight
        nx.draw_networkx_edges(
            community_graph, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color='grey'
        )
        
        # Add labels
        nx.draw_networkx_labels(
            community_graph, pos,
            labels={comm_id: f"C{comm_id}\n({community_graph.nodes[comm_id]['size']})" for comm_id in community_graph},
            font_size=10,
            font_weight='bold'
        )
        
        plt.title("Community Interconnections", fontsize=16)
        plt.axis('off')
        
        # Save the community interconnection visualization
        interconnections_path = os.path.join(output_dir, "community_interconnections.png")
        plt.tight_layout()
        plt.savefig(interconnections_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Community interconnection visualization saved to {interconnections_path}")
        
        # --- VISUALIZATION 4: DETAILED COMMUNITY GRAPHS ---
        logger.info("Creating detailed visualizations for top communities...")
        
        # Create a directory for detailed community visualizations
        detailed_dir = os.path.join(output_dir, "detailed_communities")
        os.makedirs(detailed_dir, exist_ok=True)
        
        # Visualize top N communities individually
        top_n_detailed = min(10, len(largest_communities))  # Limit to top 10 communities
        detailed_paths = []
        
        for i, (comm_id, size) in enumerate(largest_communities[:top_n_detailed]):
            # Skip very small communities
            if size < 10:
                continue
                
            # Get nodes in this community
            comm_nodes = [node for node in communities[comm_id] if node in G_viz]
            
            # Skip if no valid nodes
            if not comm_nodes:
                continue
                
            # Create subgraph for this community
            subgraph = G_viz.subgraph(comm_nodes).copy()
            
            # Skip if too few edges
            if subgraph.number_of_edges() < 5:
                continue
                
            # Compute positions - try to use a layout that works well for the graph's size
            if subgraph.number_of_nodes() < 100:
                try:
                    pos = nx.spring_layout(subgraph, seed=42)
                except:
                    pos = nx.random_layout(subgraph)
            else:
                # For larger communities, use faster layout algorithms
                try:
                    pos = nx.kamada_kawai_layout(subgraph)
                except:
                    pos = nx.random_layout(subgraph)
            
            # Create figure
            plt.figure(figsize=(12, 12))
            
            # Calculate node sizes based on degree
            node_degrees = dict(subgraph.degree())
            max_degree = max(node_degrees.values()) if node_degrees else 1
            node_sizes = [20 + 200 * (node_degrees[n] / max_degree) for n in subgraph.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                subgraph, pos,
                node_size=node_sizes,
                node_color=color_map(comm_to_color.get(comm_id, 0) % 20),
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph, pos,
                alpha=0.2,
                width=0.5
            )
            
            # Add title and labels
            plt.title(f"Community {comm_id} - {size} nodes, {subgraph.number_of_edges()} edges", fontsize=16)
            plt.axis('off')
            
            # Save the detailed community visualization
            detailed_path = os.path.join(detailed_dir, f"community_{comm_id}_detail.png")
            plt.tight_layout()
            plt.savefig(detailed_path, dpi=200, bbox_inches="tight")
            plt.close()
            
            detailed_paths.append((comm_id, detailed_path))
            logger.info(f"Detailed visualization for community {comm_id} saved to {detailed_path}")
        
        # Create HTML report
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
                .warning {{ background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .image-row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .image-container {{ width: 45%; min-width: 300px; margin-bottom: 20px; }}
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
                <p>Graph connected components: {n_components}</p>
                <p>Largest connected component: {largest_cc_size} nodes ({largest_cc_size/G.number_of_nodes()*100:.1f}% of graph)</p>
            </div>
            """
            
        # Add warning if the graph is highly disconnected
        if n_components > 10 and n_components > len(communities) * 0.5:
            html_content += f"""
            <div class="warning">
                <strong>Note:</strong> This graph has {n_components} connected components, which may naturally form separate communities.
                High modularity values are expected in disconnected graphs as there are no edges between components.
            </div>
            """
        
        html_content += f"""
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
        """
        
        # Add detailed community visualizations
        if detailed_paths:
            html_content += f"""
            <div class="section">
                <h2>Detailed Community Visualizations</h2>
                <div class="image-row">
            """
            
            for comm_id, path in detailed_paths:
                rel_path = os.path.relpath(path, output_dir)
                html_content += f"""
                <div class="image-container">
                    <h3>Community {comm_id}</h3>
                    <img src="{rel_path}" alt="Community {comm_id} Detail">
                </div>
                """
                
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML report
        html_path = os.path.join(output_dir, "community_report.html")
        try:
            with open(html_path, 'w') as f:
                f.write(html_content)
        except Exception as e:
            logger.error(f"Error writing HTML report: {str(e)}")
            # Try simpler path as fallback
            html_path = "community_report.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
        
        elapsed = time.time() - start_time
        logger.info(f"Community visualization completed in {elapsed:.2f} seconds")
        logger.info(f"Comprehensive visualization report saved to {html_path}")
        
        return html_path
    
    except Exception as e:
        logger.error(f"Error in community visualization: {str(e)}")
        return None

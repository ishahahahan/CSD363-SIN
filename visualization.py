"""
Visualization utilities for the Hybrid Community Detection project.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import to_rgba

from utils import sample_graph, community_to_node_map, add_community_attributes

def plot_communities(graph, communities, title="Community Structure", output_file=None, 
                    max_nodes=500, layout=None, node_size=50, edge_alpha=0.1, figsize=(12, 10)):
    """
    Visualize the communities in a graph.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    title : str, optional
        Plot title
    output_file : str, optional
        Path to save the plot
    max_nodes : int, optional
        Maximum number of nodes to display
    layout : dict, optional
        Pre-computed node positions
    node_size : int, optional
        Size of nodes in the visualization
    edge_alpha : float, optional
        Transparency of edges
    figsize : tuple, optional
        Figure size
    """
    # Sample the graph if it's too large
    if graph.number_of_nodes() > max_nodes:
        print(f"Graph is too large ({graph.number_of_nodes()} nodes). Sampling {max_nodes} nodes.")
        sampled_graph = sample_graph(graph, max_nodes=max_nodes)
        
        # Adjust communities to match the sampled graph
        sampled_communities = {}
        for comm_id, nodes in communities.items():
            sampled_nodes = [n for n in nodes if n in sampled_graph]
            if sampled_nodes:
                sampled_communities[comm_id] = sampled_nodes
        
        graph = sampled_graph
        communities = sampled_communities
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Generate node positions if not provided
    if layout is None:
        print("Computing layout...")
        try:
            pos = nx.spring_layout(graph, seed=42)
        except:
            # Fall back to simpler layout for large graphs
            pos = nx.kamada_kawai_layout(graph)
    else:
        pos = layout
    
    # Map nodes to communities
    node_to_community = community_to_node_map(communities)
    
    # Get list of unique communities
    unique_communities = sorted(set(node_to_community.values()))
    
    # Generate colors for communities
    colors = cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    color_map = {comm_id: colors[i] for i, comm_id in enumerate(unique_communities)}
    
    # Draw edges with low alpha for better visibility
    nx.draw_networkx_edges(graph, pos, alpha=edge_alpha, edge_color='gray')
    
    # Draw nodes community by community
    for comm_id, nodes in communities.items():
        color = color_map.get(comm_id, 'gray')
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[n for n in nodes if n in graph],  # Ensure nodes exist in graph
            node_color=[to_rgba(color)],  # Convert to rgba for consistent format
            node_size=node_size,
            label=f"Community {comm_id}"
        )
    
    # Add legend if there aren't too many communities
    if len(communities) <= 20:  # Limit legend to 20 entries for readability
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Communities")
    
    plt.title(title)
    plt.axis('off')
    
    # Save or display the plot
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_comparison(communities_dict, execution_times, output_file=None, metrics=None):
    """
    Plot comparative metrics for different community detection algorithms.
    
    Parameters:
    -----------
    communities_dict : dict
        Dictionary mapping algorithm names to community structures
    execution_times : dict
        Dictionary mapping algorithm names to execution times
    output_file : str, optional
        Path to save the plot
    metrics : dict, optional
        Dictionary mapping algorithm names to metrics
    """
    # Set up the figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Number of communities and sizes
    ax = axs[0]
    algorithms = list(communities_dict.keys())
    
    # Calculate metrics
    num_communities = [len(comm) for comm in communities_dict.values()]
    avg_size = [sum(len(nodes) for nodes in comm.values()) / len(comm) if len(comm) > 0 else 0 
               for comm in communities_dict.values()]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, num_communities, width, label='Number of Communities')
    ax.set_ylabel('Number of Communities')
    
    # Create a second y-axis for average community size
    ax2 = ax.twinx()
    ax2.bar(x + width/2, avg_size, width, color='orange', label='Avg Community Size')
    ax2.set_ylabel('Average Community Size')
    
    # Set x-axis properties
    ax.set_xticks(x)
    ax.set_xticklabels([algo.title() for algo in algorithms])
    
    # Add legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    
    ax.set_title('Community Structure Comparison')
    
    # Plot 2: Execution times
    ax = axs[1]
    
    # Get execution times
    times = [execution_times.get(algo, 0) for algo in algorithms]
    
    # Create bars
    ax.bar(x, times, width=0.6)
    
    # Set axis properties
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels([algo.title() for algo in algorithms])
    
    ax.set_title('Execution Time Comparison')
    
    # Add time values as text on the bars
    for i, v in enumerate(times):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha='center', va='bottom')
    
    # Set layout and save/show plot
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_community_sizes(communities, title="Community Size Distribution", output_file=None):
    """
    Plot the distribution of community sizes.
    
    Parameters:
    -----------
    communities : dict
        Dictionary mapping community ID to list of nodes
    title : str, optional
        Plot title
    output_file : str, optional
        Path to save the plot
    """
    # Get community sizes
    sizes = [len(nodes) for nodes in communities.values()]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(sizes, kde=True, bins=30)
    
    # Add statistics
    plt.axvline(np.mean(sizes), color='r', linestyle='--', label=f'Mean: {np.mean(sizes):.1f}')
    plt.axvline(np.median(sizes), color='g', linestyle='-.', label=f'Median: {np.median(sizes):.1f}')
    
    # Set labels and title
    plt.xlabel('Community Size (number of nodes)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    
    # Save or display the plot
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved community size distribution to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_metrics_comparison(metrics_dict, output_file=None):
    """
    Plot comparative metrics for different community detection algorithms.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary mapping algorithm names to metric dictionaries
    output_file : str, optional
        Path to save the plot
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Extract algorithm names and metric keys
    algorithms = list(metrics_dict.keys())
    
    # Common metrics to compare
    common_metrics = ['modularity', 'avg_conductance', 'coverage']
    if 'nmi' in next(iter(metrics_dict.values())):
        common_metrics.append('nmi')
    
    # Number of metrics and algorithms
    n_metrics = len(common_metrics)
    n_algos = len(algorithms)
    
    # Set up the plot grid
    x = np.arange(n_algos)
    width = 0.8 / n_metrics
    
    # Plot each metric as a group of bars
    for i, metric in enumerate(common_metrics):
        values = [metrics_dict[algo].get(metric, 0) for algo in algorithms]
        plt.bar(x + i*width - (n_metrics-1)*width/2, values, width, label=metric.title())
    
    # Set up the plot appearance
    plt.xlabel('Algorithm')
    plt.ylabel('Score')
    plt.title('Metrics Comparison Across Algorithms')
    plt.xticks(x, [algo.title() for algo in algorithms])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save or display the plot
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {output_file}")
    else:
        plt.show()
    
    plt.close()

def save_community_graph(graph, communities, output_file):
    """
    Save a graph with community attributes to file for external analysis.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    output_file : str
        Path to save the graph
    """
    # Add community IDs as node attributes
    graph_with_attrs = add_community_attributes(graph, communities)
    
    # Save the graph
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nx.write_gexf(graph_with_attrs, output_file)
    print(f"Saved community graph to {output_file}")

def plot_hierarchy(communities_dict, output_file=None, figsize=(12, 8)):
    """
    Visualize the hierarchical relationship between communities.
    
    Parameters:
    -----------
    communities_dict : dict
        Dictionary of community structures at different levels
    output_file : str, optional
        Path to save the plot
    figsize : tuple, optional
        Figure size
    """
    if len(communities_dict) < 2:
        print("Need at least two levels of communities to visualize hierarchy")
        return
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Sort levels by key
    sorted_levels = sorted(communities_dict.keys())
    
    # Calculate positions
    level_positions = {}
    y_pos = 0
    
    for level in sorted_levels:
        communities = communities_dict[level]
        n_communities = len(communities)
        
        # Calculate x positions for communities
        x_positions = np.linspace(0, 1, n_communities + 2)[1:-1]  # Exclude endpoints
        
        # Store positions
        level_positions[level] = {
            'y': y_pos,
            'x': x_positions,
            'communities': communities
        }
        
        # Update y position for next level
        y_pos += 1
    
    # Draw levels and communities
    for i, level in enumerate(sorted_levels):
        level_info = level_positions[level]
        y = level_info['y']
        
        # Draw level line
        plt.axhline(y, color='black', linestyle='-', alpha=0.3)
        
        # Draw communities as nodes
        for j, x in enumerate(level_info['x']):
            plt.scatter(x, y, s=100, color=plt.cm.tab20(j % 20), edgecolor='black', zorder=3)
            plt.text(x, y - 0.05, f"{j}", ha='center', va='top', fontsize=8)
    
    # Draw connections between levels
    for i in range(len(sorted_levels) - 1):
        level1 = sorted_levels[i]
        level2 = sorted_levels[i+1]
        
        level1_info = level_positions[level1]
        level2_info = level_positions[level2]
        
        # Create node-to-community mappings
        node_to_comm1 = community_to_node_map(level1_info['communities'])
        node_to_comm2 = community_to_node_map(level2_info['communities'])
        
        # Find overlaps between communities
        connections = defaultdict(int)
        
        for node, comm2_id in node_to_comm2.items():
            if node in node_to_comm1:
                comm1_id = node_to_comm1[node]
                connections[(comm1_id, comm2_id)] += 1
        
        # Draw connections
        for (comm1_id, comm2_id), count in connections.items():
            # Get indices
            idx1 = list(level1_info['communities'].keys()).index(comm1_id)
            idx2 = list(level2_info['communities'].keys()).index(comm2_id)
            
            # Get positions

            x1 = level1_info['x'][idx1]
            x2 = level2_info['x'][idx2]
            y1 = level1_info['y']
            y2 = level2_info['y']
            
            # Draw line with width proportional to overlap
            width = np.log1p(count) / 2
            alpha = min(0.8, width / 2)
            plt.plot([x1, x2], [y1, y2], color='gray', linewidth=width, alpha=alpha, zorder=1)
    
    # Set up plot appearance
    plt.ylim(-0.5, len(sorted_levels) - 0.5)
    plt.xlim(-0.1, 1.1)
    plt.title('Community Hierarchy')
    plt.axis('off')
    
    # Save or display the plot
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved hierarchy plot to {output_file}")
    else:
        plt.show()
    
    plt.close()
    
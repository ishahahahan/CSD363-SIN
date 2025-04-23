import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import logging

logger = logging.getLogger('community_pipeline')

def plot_algorithm_metrics(tracked_metrics, algorithm_type=None, save_path=None):
    """
    Plot metrics tracked over algorithm steps
    
    Args:
        tracked_metrics (dict): Dictionary of tracked metrics
        algorithm_type (str): 'girvan_newman', 'infomap', or None
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not tracked_metrics:
        logger.error("No metrics to plot")
        return None
        
    steps = tracked_metrics.get('steps', 0)
    if steps <= 1:
        logger.warning("Not enough steps to create a meaningful plot")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Algorithm Metrics Over Time ({algorithm_type if algorithm_type else 'Unknown'})", 
                fontsize=16)
    
    # Plot basic metrics
    x = range(1, len(tracked_metrics.get('modularity', [])) + 1)
    
    # Plot modularity
    ax = axes[0, 0]
    ax.plot(x, tracked_metrics.get('modularity', []), 'b-', marker='o', markersize=3)
    ax.set_title('Modularity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Modularity')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot conductance
    ax = axes[0, 1]
    ax.plot(x, tracked_metrics.get('conductance', []), 'r-', marker='o', markersize=3)
    ax.set_title('Conductance')
    ax.set_xlabel('Step')
    ax.set_ylabel('Conductance')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot number of communities
    ax = axes[1, 0]
    ax.plot(x, tracked_metrics.get('num_communities', []), 'g-', marker='o', markersize=3)
    ax.set_title('Number of Communities')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot algorithm-specific metrics
    ax = axes[1, 1]
    if algorithm_type == 'girvan_newman' and 'edge_betweenness' in tracked_metrics:
        ax.plot(x, tracked_metrics.get('edge_betweenness', []), 'c-', marker='o', markersize=3)
        ax.set_title('Edge Betweenness')
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg. Edge Betweenness')
    elif algorithm_type == 'infomap' and 'description_length' in tracked_metrics:
        ax.plot(x, tracked_metrics.get('description_length', []), 'm-', marker='o', markersize=3)
        ax.set_title('Description Length')
        ax.set_xlabel('Step')
        ax.set_ylabel('Description Length')
    else:
        # Plot coverage as default
        ax.plot(x, tracked_metrics.get('coverage', []), 'y-', marker='o', markersize=3)
        ax.set_title('Coverage')
        ax.set_xlabel('Step')
        ax.set_ylabel('Coverage')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig

def plot_edge_betweenness_distribution(G, edge_betweenness, partition=None, top_n=20, save_path=None):
    """
    Plot the distribution of edge betweenness values
    
    Args:
        G (networkx.Graph): Input graph
        edge_betweenness (dict): Dictionary of edge betweenness values
        partition (dict, optional): Community partition
        top_n (int): Number of top edges to highlight
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not edge_betweenness:
        logger.error("No edge betweenness data to plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Edge Betweenness Analysis', fontsize=16)
    
    # Plot distribution histogram
    values = list(edge_betweenness.values())
    sns.histplot(values, kde=True, ax=ax1)
    ax1.set_title('Distribution of Edge Betweenness')
    ax1.set_xlabel('Edge Betweenness Value')
    ax1.set_ylabel('Frequency')
    
    # Sort edges by betweenness
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top edges by betweenness
    top_edges = sorted_edges[:top_n]
    edges, values = zip(*top_edges) if top_edges else ([], [])
    
    # Get edge labels
    edge_labels = [f"{u}-{v}" for (u, v) in edges]
    
    # Color by whether they're inter-community edges
    colors = []
    if partition:
        for u, v in edges:
            if u in partition and v in partition and partition[u] != partition[v]:
                colors.append('red')  # Inter-community edge
            else:
                colors.append('blue')  # Intra-community edge
    else:
        colors = ['blue'] * len(edges)
    
    ax2.barh(range(len(edges)), values, color=colors)
    ax2.set_yticks(range(len(edges)))
    ax2.set_yticklabels(edge_labels, fontsize=8)
    ax2.set_title(f'Top {top_n} Edges by Betweenness')
    ax2.set_xlabel('Edge Betweenness Value')
    
    # Add a legend
    if partition:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Inter-community Edge'),
            Patch(facecolor='blue', label='Intra-community Edge')
        ]
        ax2.legend(handles=legend_elements)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig

def plot_description_length_contribution(G, communities, community_entropy, save_path=None):
    """
    Plot the contribution of each community to the description length
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Communities dict
        community_entropy (dict): Entropy contribution of each community
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not community_entropy:
        logger.error("No community entropy data to plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('InfoMap Description Length Analysis', fontsize=16)
    
    # Plot entropy contributions
    comm_ids = list(community_entropy.keys())
    entropy_values = list(community_entropy.values())
    
    # Sort by entropy contribution
    sorted_indices = np.argsort(entropy_values)[::-1]  # Descending order
    sorted_ids = [comm_ids[i] for i in sorted_indices]
    sorted_values = [entropy_values[i] for i in sorted_indices]
    
    # Plot top communities
    top_n = min(20, len(sorted_ids))
    ax1.barh(range(top_n), sorted_values[:top_n])
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f"Comm {sorted_ids[i]}" for i in range(top_n)], fontsize=8)
    ax1.set_title(f'Top {top_n} Communities by Entropy Contribution')
    ax1.set_xlabel('Entropy Contribution')
    
    # Plot relationship between community size and entropy
    sizes = []
    entropies = []
    for comm_id, entropy in community_entropy.items():
        if comm_id in communities:
            sizes.append(len(communities[comm_id]))
            entropies.append(entropy)
    
    ax2.scatter(sizes, entropies, alpha=0.7)
    ax2.set_title('Community Size vs. Entropy Contribution')
    ax2.set_xlabel('Community Size')
    ax2.set_ylabel('Entropy Contribution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(sizes) > 1:
        z = np.polyfit(sizes, entropies, 1)
        p = np.poly1d(z)
        ax2.plot(sizes, p(sizes), "r--", alpha=0.8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig

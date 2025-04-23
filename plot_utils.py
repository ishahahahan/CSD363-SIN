"""
Utility functions for plotting and visualization
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging
import os

logger = logging.getLogger('community_pipeline')

def plot_community_summary(G, communities, output_dir='results'):
    """
    Generate a summary visualization of community structure and metrics
    
    Args:
        G (networkx.Graph): Input graph
        communities (dict): Dictionary mapping community IDs to node lists
        output_dir (str): Directory to save output files
        
    Returns:
        str: Path to output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get community sizes
    sizes = [len(nodes) for nodes in communities.values()]
    
    # Plot size distribution
    ax1.hist(sizes, bins=20, alpha=0.7)
    ax1.set_title('Community Size Distribution')
    ax1.set_xlabel('Community Size')
    ax1.set_ylabel('Frequency')
    
    # Plot connectivity analysis 
    # Get connected components
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    
    # Get top 10 component sizes
    top_components = sorted(component_sizes, reverse=True)[:10]
    
    ax2.bar(range(len(top_components)), top_components)
    ax2.set_title('Top 10 Connected Components')
    ax2.set_xlabel('Component Rank')
    ax2.set_ylabel('Size')
    
    # Add text about disconnectivity
    if len(components) > 1:
        fig.text(0.5, 0.01, 
                f"Graph has {len(components)} connected components. This significantly affects community detection.",
                ha='center', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'community_summary.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_connectivity_report(G, output_path='connectivity_report.png'):
    """
    Generate a detailed report on graph connectivity
    
    Args:
        G (networkx.Graph): Input graph
        output_path (str): Path to save the output file
        
    Returns:
        str: Path to output file
    """
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution of component sizes
    ax1.hist(component_sizes, bins=20, color='skyblue', alpha=0.7)
    ax1.set_title(f'Distribution of {len(components)} Connected Components')
    ax1.set_xlabel('Component Size')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative size distribution
    sorted_sizes = sorted(component_sizes, reverse=True)
    cumsum = np.cumsum(sorted_sizes) / sum(sorted_sizes)
    ax2.plot(range(len(sorted_sizes)), cumsum, marker='o', markersize=3)
    ax2.set_title('Cumulative Node Coverage')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Fraction of Nodes Covered')
    ax2.grid(True, alpha=0.3)
    
    # Calculate useful statistics
    largest_component = max(components, key=len)
    largest_pct = len(largest_component) / G.number_of_nodes() * 100
    
    # Add annotation with statistics
    stats_text = f"""
    Graph Statistics:
    - Total nodes: {G.number_of_nodes()}
    - Total edges: {G.number_of_edges()}
    - Connected components: {len(components)}
    - Largest component: {len(largest_component)} nodes ({largest_pct:.1f}%)
    - Density: {nx.density(G):.6f}
    """
    
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the text
    plt.savefig(output_path)
    plt.close()
    
    return output_path

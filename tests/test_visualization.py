import os
import pytest
import networkx as nx
from community_pipeline.visualization import plot_communities

@pytest.fixture
def test_graph_with_communities():
    """Create a test graph with predefined communities"""
    # Create a graph with two clear communities
    G = nx.random_partition_graph([8, 12], 0.7, 0.1, seed=42)
    
    # Define communities
    communities = {0: [], 1: []}
    
    for i, node in enumerate(G.nodes()):
        comm_id = 0 if i < 8 else 1
        communities[comm_id].append(node)
    
    return G, communities

def test_plot_communities(test_graph_with_communities):
    """Test community visualization"""
    G, communities = test_graph_with_communities
    
    # Run visualization
    output_path = plot_communities(G, communities, max_nodes=50)
    
    # Check return value
    assert isinstance(output_path, str)
    
    # Check if file was created
    assert os.path.exists(output_path)
    
    # Clean up
    os.remove(output_path)

def test_plot_communities_large_graph():
    """Test visualization with sampling for larger graphs"""
    # Create a larger graph
    G = nx.random_partition_graph([50, 70, 80], 0.7, 0.05, seed=42)
    
    # Define communities
    communities = {0: [], 1: [], 2: []}
    
    for i, node in enumerate(G.nodes()):
        if i < 50:
            comm_id = 0
        elif i < 120:
            comm_id = 1
        else:
            comm_id = 2
        communities[comm_id].append(node)
    
    # Run visualization with sampling
    output_path = plot_communities(G, communities, max_nodes=100)
    
    # Check if file was created
    assert os.path.exists(output_path)
    
    # Clean up
    os.remove(output_path)

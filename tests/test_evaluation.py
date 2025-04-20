import pytest
import networkx as nx
import numpy as np
from community_pipeline.evaluation import (
    compute_modularity, 
    compute_conductance, 
    compute_nmi, 
    evaluate_all
)

@pytest.fixture
def test_graph_with_communities():
    """Create a test graph with predefined communities"""
    # Create a graph with two clear communities
    G = nx.random_partition_graph([8, 12], 0.7, 0.1, seed=42)
    
    # Define partition (ground truth)
    partition = {}
    communities = {0: [], 1: []}
    
    for i, node in enumerate(G.nodes()):
        comm_id = 0 if i < 8 else 1
        partition[node] = comm_id
        communities[comm_id].append(node)
    
    return G, partition, communities

def test_compute_modularity(test_graph_with_communities):
    """Test modularity computation"""
    G, partition, _ = test_graph_with_communities
    
    modularity = compute_modularity(G, partition)
    
    # Check return type
    assert isinstance(modularity, float)
    
    # Modularity should be between -0.5 and 1.0 for reasonable partitions
    assert -0.5 <= modularity <= 1.0

def test_compute_conductance(test_graph_with_communities):
    """Test conductance computation"""
    G, _, communities = test_graph_with_communities
    
    conductance_values, avg_conductance = compute_conductance(G, communities)
    
    # Check return types
    assert isinstance(conductance_values, list)
    assert isinstance(avg_conductance, float)
    
    # Conductance should be between 0 and 1
    assert all(0 <= c <= 1 for c in conductance_values)
    assert 0 <= avg_conductance <= 1

def test_compute_nmi(test_graph_with_communities):
    """Test NMI computation"""
    _, partition, _ = test_graph_with_communities
    
    # Create a slightly perturbed partition as a prediction
    pred_partition = partition.copy()
    
    # Flip some assignments to make it imperfect
    for i, (node, _) in enumerate(partition.items()):
        if i % 5 == 0:  # Flip every 5th node
            pred_partition[node] = 1 - partition[node]  # Flip between 0 and 1
    
    nmi = compute_nmi(pred_partition, partition)
    
    # Check return type
    assert isinstance(nmi, float)
    
    # NMI should be between 0 and 1
    assert 0 <= nmi <= 1
    
    # Since partitions are similar but not identical, NMI should be high but not 1
    assert 0.5 < nmi < 1.0
    
    # Test with identical partitions (should be 1.0)
    perfect_nmi = compute_nmi(partition, partition)
    assert np.isclose(perfect_nmi, 1.0)

def test_evaluate_all(test_graph_with_communities):
    """Test the full evaluation"""
    G, partition, _ = test_graph_with_communities
    
    metrics = evaluate_all(G, partition)
    
    # Check return type
    assert isinstance(metrics, dict)
    
    # Check required keys
    required_keys = ['modularity', 'conductance_values', 'avg_conductance', 
                     'num_communities', 'community_sizes']
    for key in required_keys:
        assert key in metrics
    
    # Test with ground truth
    metrics_with_gt = evaluate_all(G, partition, ground_truth=partition)
    assert 'nmi' in metrics_with_gt
    assert np.isclose(metrics_with_gt['nmi'], 1.0)

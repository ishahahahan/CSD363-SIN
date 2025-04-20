import pytest
import networkx as nx
from community_pipeline.detection import run_louvain, refine_girvan_newman, enhance_infomap

@pytest.fixture
def test_graph():
    """Create a test graph with community structure"""
    # Create a graph with two clear communities
    G = nx.random_partition_graph([10, 15], 0.8, 0.05, seed=42)
    return G

def test_run_louvain(test_graph):
    """Test Louvain community detection"""
    partition, communities = run_louvain(test_graph)
    
    # Check return types
    assert isinstance(partition, dict)
    assert isinstance(communities, dict)
    
    # Check if all nodes are assigned to communities
    assert len(partition) == test_graph.number_of_nodes()
    
    # Check if communities dictionary is non-empty
    assert len(communities) > 0
    
    # Check that communities are disjoint
    all_nodes = []
    for comm_nodes in communities.values():
        all_nodes.extend(comm_nodes)
    assert len(all_nodes) == len(set(all_nodes))

def test_refine_girvan_newman(test_graph):
    """Test Girvan-Newman refinement"""
    # First run Louvain to get initial communities
    _, communities = run_louvain(test_graph)
    
    # Run refinement
    refined_partition = refine_girvan_newman(
        test_graph, 
        communities, 
        size_threshold=5,  # Small threshold for test
        target_subcommunities=2
    )
    
    # Check return type
    assert isinstance(refined_partition, dict)
    
    # Check if all nodes are assigned to communities
    assert len(refined_partition) == test_graph.number_of_nodes()

def test_enhance_infomap(test_graph):
    """Test Infomap enhancement"""
    # First run Louvain to get initial partition
    partition, communities = run_louvain(test_graph)
    
    try:
        # This test might be skipped if Infomap is not available
        enhanced_partition = enhance_infomap(
            test_graph,
            partition,
            communities,
            modularity_threshold=0.9  # High threshold to force enhancement
        )
        
        # Check return type
        assert isinstance(enhanced_partition, dict)
        
        # Check if all nodes are assigned to communities
        assert len(enhanced_partition) == test_graph.number_of_nodes()
    except ImportError:
        pytest.skip("Infomap not available, skipping test")

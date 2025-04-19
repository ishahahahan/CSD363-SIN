import os
import pytest
import networkx as nx
from community_pipeline.data_io import download_and_extract, load_graph, get_graph

@pytest.fixture
def temp_data_dir(tmpdir):
    """Create a temporary directory for test data"""
    data_dir = tmpdir.mkdir("test_data")
    return str(data_dir)

def test_load_graph():
    """Test loading a graph from an edge list"""
    # Create a simple test graph
    G = nx.karate_club_graph()
    edge_file = "test_edges.txt"
    
    # Write edge list to file
    with open(edge_file, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    # Test loading the graph
    loaded_G = load_graph(edge_file)
    
    # Check graph properties
    assert isinstance(loaded_G, nx.Graph)
    assert loaded_G.number_of_nodes() == G.number_of_nodes()
    assert loaded_G.number_of_edges() == G.number_of_edges()
    
    # Test sampling
    sampled_G = load_graph(edge_file, sample_size=10)
    assert isinstance(sampled_G, nx.Graph)
    assert sampled_G.number_of_edges() <= 10
    
    # Clean up
    os.remove(edge_file)

def test_get_graph(temp_data_dir, monkeypatch):
    """Test the graph caching mechanism"""
    # Create a mock graph for testing
    def mock_download(*args, **kwargs):
        edge_file = os.path.join(temp_data_dir, "test_edges.txt")
        G = nx.karate_club_graph()
        with open(edge_file, "w") as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        return edge_file
    
    # Patch the download function
    monkeypatch.setattr("community_pipeline.data_io.download_and_extract", mock_download)
    
    # Test getting graph
    G = get_graph(temp_data_dir, sample_size=None)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    
    # Check if pickle was created
    pickle_path = os.path.join(temp_data_dir, "livejournal.pkl")
    assert os.path.exists(pickle_path)
    
    # Test loading from pickle
    G2 = get_graph(temp_data_dir, sample_size=None)
    assert isinstance(G2, nx.Graph)
    assert G2.number_of_nodes() == G.number_of_nodes()

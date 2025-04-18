"""
Utility functions for the Hybrid Community Detection project.
"""

import os
import json
import time
import logging
import networkx as nx
import numpy as np
from contextlib import contextmanager
from datetime import datetime

def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_file : str, optional
        Path to the log file
    level : int, optional
        Logging level
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('hybrid_community_detection')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

@contextmanager
def timer(message=None):
    """
    Context manager for timing code execution.
    
    Parameters:
    -----------
    message : str, optional
        Message to print before timing
    
    Yields:
    -------
    TimerContextManager
        Timer object with elapsed property
    """
    class TimerContextManager:
        def __init__(self):
            self.elapsed = 0
    
    timer_obj = TimerContextManager()
    
    if message:
        print(f"{message}...")
    
    start_time = time.time()
    yield timer_obj
    timer_obj.elapsed = time.time() - start_time
    
    if message:
        print(f"{message} completed in {timer_obj.elapsed:.4f} seconds")

def save_results(results, file_path):
    """
    Save experiment results to a JSON file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    file_path : str
        Path to save the results
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(file_path):
    """
    Load experiment results from a JSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to the results file
    
    Returns:
    --------
    dict
        Results dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def community_to_node_map(communities):
    """
    Convert community structure from community->nodes to node->community.
    
    Parameters:
    -----------
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    dict
        Dictionary mapping node ID to community ID
    """
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = comm_id
    return node_to_community

def sample_graph(graph, max_nodes=1000, seed=None):
    """
    Sample a subgraph for visualization purposes.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    max_nodes : int, optional
        Maximum number of nodes in the sample
    seed : int, optional
        Random seed
    
    Returns:
    --------
    networkx.Graph
        Sampled subgraph
    """
    if graph.number_of_nodes() <= max_nodes:
        return graph
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Sample nodes
    sampled_nodes = list(np.random.choice(
        list(graph.nodes()), 
        size=max_nodes, 
        replace=False
    ))
    
    # Extract the subgraph
    subgraph = graph.subgraph(sampled_nodes).copy()
    
    # Keep only the largest connected component
    if not nx.is_connected(subgraph):
        largest_cc = max(nx.connected_components(subgraph), key=len)
        subgraph = subgraph.subgraph(largest_cc).copy()
    
    return subgraph

def get_community_subgraph(graph, communities, comm_id):
    """
    Extract a subgraph for a specific community.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    comm_id : int
        Community ID
    
    Returns:
    --------
    networkx.Graph
        Community subgraph
    """
    if comm_id not in communities:
        return nx.Graph()
    
    return graph.subgraph(communities[comm_id]).copy()

def get_timestamp():
    """
    Get a formatted timestamp string.
    
    Returns:
    --------
    str
        Timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def add_community_attributes(graph, communities):
    """
    Add community IDs as node attributes in the graph.
    
    Parameters:
    -----------
    graph : networkx.Graph
        Input graph
    communities : dict
        Dictionary mapping community ID to list of nodes
    
    Returns:
    --------
    networkx.Graph
        Graph with community attributes
    """
    result = graph.copy()
    
    # Convert community structure to node->community mapping
    node_to_community = community_to_node_map(communities)
    
    # Add community as node attribute
    nx.set_node_attributes(result, node_to_community, name='community')
    
    return result

def compare_community_structures(communities1, communities2):
    """
    Compare two community structures and calculate their overlap.
    
    Parameters:
    -----------
    communities1 : dict
        First community structure
    communities2 : dict
        Second community structure
    
    Returns:
    --------
    float
        Jaccard similarity between community structures
    """
    # Convert community structures to sets of frozensets for comparison
    structure1 = {frozenset(nodes) for nodes in communities1.values()}
    structure2 = {frozenset(nodes) for nodes in communities2.values()}
    
    # Calculate Jaccard similarity
    intersection = len(structure1.intersection(structure2))
    union = len(structure1.union(structure2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def save_community_data(communities, file_path):
    """
    Save community structure to a CSV file.
    
    Parameters:
    -----------
    communities : dict
        Dictionary mapping community ID to list of nodes
    file_path : str
        Path to save the CSV file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write('node_id,community_id\n')
        for comm_id, nodes in communities.items():
            for node in nodes:
                f.write(f'{node},{comm_id}\n')
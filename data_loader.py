"""
LiveJournal dataset loader and preprocessor for community detection.
"""

import os
import gzip
import urllib.request
import random
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class LiveJournalLoader:
    """
    Load and preprocess the LiveJournal dataset from SNAP.
    """
    
    def __init__(self, data_dir="./data"):
        """
        Initialize the LiveJournal dataset loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store/load the dataset files
        """
        self.data_dir = data_dir
        self.edge_file = os.path.join(data_dir, "soc-LiveJournal1.txt.gz")
        self.comm_file = os.path.join(data_dir, "com-lj.top5000.cmty.txt.gz")
        self.url_edges = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
        self.url_communities = "https://snap.stanford.edu/data/com-lj.top5000.cmty.txt.gz"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def download_data(self):
        """
        Download the LiveJournal dataset files if they don't exist locally.
        
        Returns:
        --------
        bool
            True if files are available (downloaded or already present)
        """
        files_to_download = [
            (self.url_edges, self.edge_file),
            (self.url_communities, self.comm_file)
        ]
        
        for url, file_path in files_to_download:
            if not os.path.exists(file_path):
                print(f"Downloading {os.path.basename(file_path)} from {url}...")
                try:
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Downloaded {os.path.basename(file_path)} successfully.")
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    return False
            else:
                print(f"{os.path.basename(file_path)} already exists.")
        
        return True
    
    def _read_communities(self, max_communities=None):
        """
        Read community information from the LiveJournal communities file.
        
        Parameters:
        -----------
        max_communities : int, optional
            Maximum number of communities to read
        
        Returns:
        --------
        list
            List of communities, where each community is a set of node IDs
        """
        if not os.path.exists(self.comm_file):
            print(f"Community file {self.comm_file} not found.")
            return []
        
        communities = []
        
        with gzip.open(self.comm_file, 'rt') as f:
            for i, line in enumerate(f):
                if max_communities is not None and i >= max_communities:
                    break
                
                # Each line contains node IDs of a community
                community = set(map(int, line.strip().split()))
                communities.append(community)
        
        print(f"Read {len(communities)} communities.")
        return communities
    
    def _read_edges(self, node_filter=None):
        """
        Read edges from the LiveJournal edge file.
        
        Parameters:
        -----------
        node_filter : set, optional
            If provided, only edges between nodes in this set will be included
        
        Returns:
        --------
        list
            List of (source, target) edge tuples
        """
        if not os.path.exists(self.edge_file):
            print(f"Edge file {self.edge_file} not found.")
            return []
        
        edges = []
        
        with gzip.open(self.edge_file, 'rt') as f:
            # Skip comment lines
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    source, target = int(parts[0]), int(parts[1])
                    
                    # If node_filter is provided, only include edges between filtered nodes
                    if node_filter is None or (source in node_filter and target in node_filter):
                        edges.append((source, target))
        
        return edges
    
    def sample_network(self, sample_size=10000, max_communities=100, seed=42):
        """
        Create a sample network from the LiveJournal dataset.
        
        Parameters:
        -----------
        sample_size : int
            Target number of nodes in the sample
        max_communities : int
            Maximum number of communities to consider
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        networkx.Graph
            Sampled graph
        dict
            Ground truth community assignments (node_id -> community_id)
        """
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Creating a sample of size ~{sample_size} nodes from LiveJournal...")
        
        # Read communities
        communities = self._read_communities(max_communities)
        
        if not communities:
            print("No communities found. Using a synthetic network instead.")
            return self._generate_synthetic_network(sample_size)
        
        # Select a subset of communities to reach the target sample size
        selected_communities = []
        selected_nodes = set()
        
        # Sort communities by size (descending) to prioritize larger communities
        communities.sort(key=len, reverse=True)
        
        for community in communities:
            if len(selected_nodes) >= sample_size:
                break
            
            # If adding this community would exceed the target size by too much,
            # sample a portion of its nodes
            if len(selected_nodes) + len(community) > sample_size * 1.5:
                # Calculate how many more nodes we want
                remaining = sample_size - len(selected_nodes)
                # Sample that many nodes from the community
                sampled_nodes = set(random.sample(list(community), min(remaining, len(community))))
                selected_nodes.update(sampled_nodes)
                selected_communities.append(sampled_nodes)
            else:
                selected_nodes.update(community)
                selected_communities.append(community)
        
        print(f"Selected {len(selected_nodes)} nodes from {len(selected_communities)} communities.")
        
        # Read edges that connect nodes in our sample
        print("Reading edges for the sampled nodes...")
        edges = self._read_edges(node_filter=selected_nodes)
        
        print(f"Read {len(edges)} edges.")
        
        # Create networkx graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Remove isolated nodes
        connected_nodes = set()
        for source, target in edges:
            connected_nodes.add(source)
            connected_nodes.add(target)
        
        # Create ground truth community assignments
        ground_truth = {}
        for comm_id, community in enumerate(selected_communities):
            for node in community:
                if node in connected_nodes:  # Only include connected nodes
                    ground_truth[node] = comm_id
        
        # Create the final graph with only connected nodes
        final_graph = nx.Graph()
        final_graph.add_edges_from(edges)
        
        print(f"Final sampled graph has {final_graph.number_of_nodes()} nodes and {final_graph.number_of_edges()} edges.")
        
        return final_graph, ground_truth
    
    def _generate_synthetic_network(self, n_nodes=10000):
        """
        Generate a synthetic network with community structure if real data is unavailable.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes in the synthetic network
        
        Returns:
        --------
        networkx.Graph
            Synthetic graph with community structure
        dict
            Ground truth community assignments
        """
        print("Generating a synthetic network with community structure...")
        
        # Use LFR benchmark for generating a graph with community structure
        G = nx.LFR_benchmark_graph(
            n=n_nodes,
            tau1=3,      # Power law exponent for degree distribution
            tau2=1.5,    # Power law exponent for community size distribution
            mu=0.1,      # Mixing parameter
            average_degree=20,
            min_community=20,
            seed=42
        )
        
        # Extract ground truth communities
        ground_truth = {}
        for node in G.nodes():
            for comm in G.nodes[node]['community']:
                ground_truth[node] = comm
                break  # Just use the first community assignment for simplicity
        
        print(f"Generated synthetic graph with {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, and {len(set(ground_truth.values()))} communities.")
        
        return G, ground_truth
    
    def load_full_network(self):
        """
        Load the full LiveJournal network (warning: very large).
        
        Returns:
        --------
        networkx.Graph
            The complete LiveJournal social network
        dict
            Ground truth community assignments
        """
        print("Warning: Loading the full LiveJournal network (this may take a long time and require significant memory).")
        
        # Read all edges
        edges = self._read_edges()
        
        # Create the graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Read communities for ground truth
        communities = self._read_communities()
        
        # Create ground truth community assignments
        ground_truth = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                if node in G:  # Only include nodes that exist in the graph
                    ground_truth[node] = comm_id
        
        print(f"Loaded full graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        print(f"Ground truth available for {len(ground_truth)} nodes across {len(set(ground_truth.values()))} communities.")
        
        return G, ground_truth
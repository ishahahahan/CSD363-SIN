"""
Hybrid community detection pipeline combining Louvain, Girvan-Newman, and Infomap algorithms.
"""

import time
import networkx as nx
from collections import defaultdict

# Import algorithm modules
from algorithms import louvain_algorithm
from algorithms import girvan_newman
from algorithms import infomap_algorithm

class HybridCommunityDetector:
    """
    Hybrid community detection pipeline that combines Louvain, Girvan-Newman, and Infomap
    to leverage their complementary strengths.
    """
    
    def __init__(self, graph, ground_truth=None):
        """
        Initialize the hybrid community detector.
        
        Parameters:
        -----------
        graph : networkx.Graph
            Input graph
        ground_truth : dict, optional
            Ground truth community assignments (node_id -> community_id)
        """
        self.graph = graph
        self.ground_truth = ground_truth
        
        # Results storage
        self.louvain_communities = None
        self.refined_communities = None
        self.communities = None
        self.execution_times = {
            "louvain": 0,
            "girvan_newman": 0,
            "infomap": 0,
            "total": 0
        }
    
    def detect_communities(self):
        """
        Execute the hybrid community detection pipeline.
        
        Returns:
        --------
        dict
            Dictionary mapping community ID to list of nodes
        """
        start_time = time.time()
        
        print("\n=== Starting Hybrid Community Detection Pipeline ===")
        
        # Step 1: Apply Louvain method for initial coarse communities
        print("\nStep 1: Applying Louvain method for initial partitioning...")
        self.louvain_communities, louvain_time = louvain_algorithm.detect_communities(self.graph)
        self.execution_times["louvain"] = louvain_time
        
        # Step 2: Refine each Louvain community using Girvan-Newman
        print("\nStep 2: Refining communities with Girvan-Newman...")
        gn_start_time = time.time()
        self.refined_communities = girvan_newman.refine_communities(self.graph, self.louvain_communities)
        self.execution_times["girvan_newman"] = time.time() - gn_start_time
        
        # Step 3: Handle boundary nodes with Infomap
        print("\nStep 3: Handling boundary nodes with Infomap...")
        infomap_start_time = time.time()
        self.communities = infomap_algorithm.handle_boundary_nodes(self.graph, self.refined_communities)
        self.execution_times["infomap"] = time.time() - infomap_start_time
        
        # Get total execution time
        self.execution_times["total"] = time.time() - start_time
        
        print(f"\nHybrid pipeline complete in {self.execution_times['total']:.2f} seconds.")
        print(f"Final number of communities: {len(self.communities)}")
        
        return self.communities
    
    def get_community_sizes(self):
        """
        Get the size distribution of detected communities.
        
        Returns:
        --------
        list
            List of community sizes
        """
        if not self.communities:
            raise ValueError("Communities have not been detected yet")
        
        return [len(nodes) for nodes in self.communities.values()]
    
    def get_node_to_community(self):
        """
        Get mapping from node ID to community ID.
        
        Returns:
        --------
        dict
            Dictionary mapping node ID to community ID
        """
        if not self.communities:
            raise ValueError("Communities have not been detected yet")
        
        node_to_community = {}
        for comm_id, nodes in self.communities.items():
            for node in nodes:
                node_to_community[node] = comm_id
        
        return node_to_community
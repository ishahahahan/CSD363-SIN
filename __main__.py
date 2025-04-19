import os
import sys
import argparse
import logging
import time
import random
import yaml
import json
import numpy as np
from collections import defaultdict

# Import pipeline modules
from community_pipeline.data_io import get_graph
from community_pipeline.detection import run_louvain, refine_girvan_newman, enhance_infomap
from community_pipeline.evaluation import evaluate_all
from community_pipeline.visualization import plot_communities

# Configure logging
def setup_logging():
    """Configure logging to write to console and file"""
    logger = logging.getLogger('community_pipeline')
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler (DEBUG level)
    file_handler = logging.FileHandler('pipeline.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_file):
    """Load configuration from YAML or JSON file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_file.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Community Detection Pipeline for LiveJournal Dataset')
    
    parser.add_argument('--data-dir', default='data',
                        help='Directory to store/find data (default: data)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for graph (default: None, use full graph)')
    parser.add_argument('--size-threshold', type=int, default=1000,
                        help='Size threshold for community refinement (default: 1000)')
    parser.add_argument('--target-subcommunities', type=int, default=5,
                        help='Target number of subcommunities for refinement (default: 5)')
    parser.add_argument('--modularity-threshold', type=float, default=0.3,
                        help='Modularity threshold for Infomap enhancement (default: 0.3)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (YAML or JSON)')
    
    return parser.parse_args()

def main():
    """Main entry point for the community detection pipeline"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Community Detection Pipeline")
    
    # Get start time for total runtime calculation
    pipeline_start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    
    # Override config with command line arguments
    data_dir = args.data_dir if hasattr(args, 'data_dir') else config.get('data_dir', 'data')
    sample_size = args.sample_size if hasattr(args, 'sample_size') else config.get('sample_size', None)
    size_threshold = args.size_threshold if hasattr(args, 'size_threshold') else config.get('size_threshold', 1000)
    target_subcommunities = args.target_subcommunities if hasattr(args, 'target_subcommunities') else config.get('target_subcommunities', 5)
    modularity_threshold = args.modularity_threshold if hasattr(args, 'modularity_threshold') else config.get('modularity_threshold', 0.3)
    
    logger.info(f"Configuration: data_dir={data_dir}, sample_size={sample_size}, "
               f"size_threshold={size_threshold}, target_subcommunities={target_subcommunities}, "
               f"modularity_threshold={modularity_threshold}")
    
    # Step 1: Load the graph
    url = "https://snap.stanford.edu/data/com-LiveJournal.tar.gz"
    G = get_graph(data_dir, url, sample_size)
    
    # Step 2: Run initial community detection using Louvain
    partition, communities = run_louvain(G)
    logger.info(f"Initial Louvain: {len(communities)} communities")
    
    # Step 3: Refine large communities using Girvan-Newman
    refined_partition = refine_girvan_newman(G, communities, size_threshold, target_subcommunities)
    
    # Update communities mapping after refinement
    refined_communities = defaultdict(list)
    for node, comm_id in refined_partition.items():
        refined_communities[comm_id].append(node)
    logger.info(f"After refinement: {len(refined_communities)} communities")
    
    # Step 4: Enhance low-modularity communities using Infomap
    final_partition = enhance_infomap(G, refined_partition, refined_communities, modularity_threshold)
    
    # Update communities mapping for final partition
    final_communities = defaultdict(list)
    for node, comm_id in final_partition.items():
        final_communities[comm_id].append(node)
    logger.info(f"Final communities: {len(final_communities)} communities")
    
    # Step 5: Evaluate results
    metrics = evaluate_all(G, final_partition)
    
    # Step 6: Visualize communities
    visualization_path = plot_communities(G, final_communities)
    
    # Print summary results
    total_runtime = time.time() - pipeline_start_time
    logger.info("=" * 50)
    logger.info("COMMUNITY DETECTION PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Nodes: {G.number_of_nodes()}")
    logger.info(f"Edges: {G.number_of_edges()}")
    logger.info(f"Number of communities: {len(final_communities)}")
    logger.info(f"Modularity: {metrics['modularity']:.4f}")
    logger.info(f"Average conductance: {metrics['avg_conductance']:.4f}")
    if metrics['nmi'] is not None:
        logger.info(f"NMI: {metrics['nmi']:.4f}")
    logger.info(f"Visualization saved to: {visualization_path}")
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()

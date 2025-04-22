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
import matplotlib.pyplot as plt

# Import pipeline modules
from data_io import get_graph
from detection import run_louvain, refine_girvan_newman, enhance_infomap
from evaluation import evaluate_all
from visualization import plot_communities

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

def save_metrics_to_file(all_metrics, filepath='evaluation_metrics.json'):
    """Save metrics dictionary to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger = logging.getLogger('community_pipeline')
    logger.info(f"Evaluation metrics saved to {filepath}")
    return filepath

def plot_metrics_comparison(metrics_dict, output_path='metrics_comparison.png'):
    """Generate a comparison plot of key metrics across pipeline stages"""
    stages = list(metrics_dict.keys())
    if 'summary' in stages:
        stages.remove('summary')  # Don't include summary in plot
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Modularity plot
    modularity_values = [metrics_dict[stage]['modularity'] for stage in stages]
    ax1.plot(stages, modularity_values, 'o-', linewidth=2, markersize=8)
    ax1.set_title('Modularity Across Pipeline Stages')
    ax1.set_ylabel('Modularity')
    ax1.grid(True)
    
    # Conductance plot
    conductance_values = [metrics_dict[stage]['avg_conductance'] for stage in stages]
    ax2.plot(stages, conductance_values, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_title('Average Conductance Across Pipeline Stages')
    ax2.set_ylabel('Avg Conductance')
    ax2.set_xlabel('Pipeline Stage')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    return output_path

# Add to your imports at the top of __main__.py
import os
import logging
import time
import networkx as nx
from collections import defaultdict

# Keep your existing setup_logging function

def load_livejournal_ground_truth(file_path):
    """Load LiveJournal ground truth communities"""
    logger = logging.getLogger('community_pipeline')
    ground_truth = {}  # node_id -> community_id
    
    try:
        with open(file_path, 'r') as f:
            for comm_id, line in enumerate(f):
                # Each line contains space-separated node IDs in one community
                nodes = line.strip().split()
                for node in nodes:
                    # Convert node IDs to integers
                    try:
                        ground_truth[int(node)] = comm_id
                    except ValueError:
                        logger.warning(f"Skipping invalid node ID: {node}")
                
                # Log progress every 1000 communities
                if comm_id > 0 and comm_id % 1000 == 0:
                    logger.info(f"Processed {comm_id} communities...")
        
        logger.info(f"Loaded ground truth for {len(ground_truth)} nodes across {comm_id+1} communities")
        return ground_truth
    
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        return {}

def load_ground_truth(data_dir, sample_size=None):
    """Load ground truth with support for different formats"""
    logger = logging.getLogger('community_pipeline')
    
    # Check for LiveJournal ground truth
    lj_ground_truth_file = os.path.join(data_dir, 'com-lj.top5000.cmty.txt')
    if os.path.exists(lj_ground_truth_file):
        logger.info(f"Loading LiveJournal ground truth from {lj_ground_truth_file}")
        return load_livejournal_ground_truth(lj_ground_truth_file)
    
    # Alternative: Check for generic ground truth format (node_id community_id)
    generic_ground_truth_file = os.path.join(data_dir, 'ground_truth.txt')
    if os.path.exists(generic_ground_truth_file):
        logger.info(f"Loading generic ground truth from {generic_ground_truth_file}")
        return load_generic_ground_truth(generic_ground_truth_file)
    
    logger.warning("No ground truth files found. NMI will not be calculated.")
    return None

def load_generic_ground_truth(file_path):
    """Load ground truth from a generic format: node_id community_id"""
    logger = logging.getLogger('community_pipeline')
    ground_truth = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):  # Skip comments and empty lines
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node_id, comm_id = parts[0], parts[1]
                        ground_truth[int(node_id)] = int(comm_id)
        
        logger.info(f"Loaded generic ground truth for {len(ground_truth)} nodes")
        return ground_truth
    
    except Exception as e:
        logger.error(f"Error loading generic ground truth: {str(e)}")
        return {}

def create_synthetic_ground_truth(G, method="louvain"):
    """Create synthetic ground truth for testing NMI when no real ground truth is available"""
    logger = logging.getLogger('community_pipeline')
    
    if method == "louvain":
        try:
            import community as community_louvain
            logger.info("Creating synthetic ground truth using Louvain algorithm")
            return community_louvain.best_partition(G)
        except Exception as e:
            logger.error(f"Error creating Louvain-based ground truth: {str(e)}")
            return {}
    
    elif method == "random":
        import random
        num_communities = max(10, G.number_of_nodes() // 100)  # At least 10 communities
        logger.info(f"Creating random synthetic ground truth with {num_communities} communities")
        return {node: random.randint(0, num_communities-1) for node in G.nodes()}
    
    else:
        logger.error(f"Unknown ground truth generation method: {method}")
        return {}

def filter_ground_truth_for_sample(ground_truth, G):
    """Filter ground truth to only include nodes present in the graph"""
    logger = logging.getLogger('community_pipeline')
    
    if not ground_truth:
        return None
    
    # Create filtered ground truth containing only nodes in G
    filtered_ground_truth = {node: comm for node, comm in ground_truth.items() if node in G}
    
    logger.info(f"Filtered ground truth from {len(ground_truth)} to {len(filtered_ground_truth)} nodes")
    
    # Check if we have enough nodes with ground truth
    if len(filtered_ground_truth) < 100:  # Arbitrary threshold
        logger.warning(f"Only {len(filtered_ground_truth)} nodes have ground truth. NMI might not be reliable.")
    
    return filtered_ground_truth

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
    data_dir = args.data_dir if args.data_dir != 'data' else config.get('data_dir', 'data')
    sample_size = args.sample_size if args.sample_size is not None else config.get('sample_size', None)
    size_threshold = args.size_threshold if args.size_threshold != 1000 else config.get('size_threshold', 1000)
    target_subcommunities = args.target_subcommunities if args.target_subcommunities != 5 else config.get('target_subcommunities', 5)
    modularity_threshold = args.modularity_threshold if args.modularity_threshold != 0.3 else config.get('modularity_threshold', 0.3)
    
    logger.info(f"Configuration: data_dir={data_dir}, sample_size={sample_size}, "
               f"size_threshold={size_threshold}, target_subcommunities={target_subcommunities}, "
               f"modularity_threshold={modularity_threshold}")
    
    # Dictionary to store metrics at each pipeline stage
    all_metrics = {}
    
    # Step 1: Load the graph
    load_start_time = time.time()
    url = "https://snap.stanford.edu/data/com-LiveJournal.tar.gz"
    edge_file_path = args.edge_file if hasattr(args, 'edge_file') else config.get('edge_file', None)
    G = get_graph(data_dir, edge_file_path=edge_file_path, url=url, sample_size=sample_size)
    load_time = time.time() - load_start_time
    logger.info(f"Graph loaded in {load_time:.2f} seconds")
    
    # Load ground truth (add this after loading the graph)
    ground_truth = None
    if config.get('use_ground_truth', True):
        ground_truth = load_ground_truth(data_dir)
        
        # If using a sampled graph, filter the ground truth
        if sample_size:
            ground_truth = filter_ground_truth_for_sample(ground_truth, G)
        
        # If no ground truth available but testing is needed, create synthetic
        if not ground_truth and config.get('use_synthetic_ground_truth', False):
            synthetic_method = config.get('synthetic_ground_truth_method', 'louvain')
            ground_truth = create_synthetic_ground_truth(G, method=synthetic_method)
            logger.info(f"Created synthetic ground truth with {len(ground_truth)} nodes")
    
    # BASELINE EVALUATION - Single Community
    baseline_start_time = time.time()
    baseline_partition = {node: 0 for node in G.nodes()}
    baseline_metrics = evaluate_all(G, baseline_partition, ground_truth)
    baseline_time = time.time() - baseline_start_time
    
    baseline_metrics['runtime'] = baseline_time
    all_metrics['baseline'] = baseline_metrics
    logger.info(f"Baseline metrics - Modularity: {baseline_metrics['modularity']:.4f}, Conductance: {baseline_metrics['avg_conductance']:.4f}")
    
    # Step 2: Run initial community detection using Louvain
    louvain_start_time = time.time()
    partition, communities = run_louvain(G)
    louvain_metrics = evaluate_all(G, partition, ground_truth)
    louvain_time = time.time() - louvain_start_time
    
    louvain_metrics['runtime'] = louvain_time
    louvain_metrics['improvement_from_baseline'] = {
        'modularity': louvain_metrics['modularity'] - baseline_metrics['modularity'],
        'conductance': baseline_metrics['avg_conductance'] - louvain_metrics['avg_conductance']
    }
    all_metrics['louvain'] = louvain_metrics
    
    logger.info(f"Initial Louvain: {len(communities)} communities")
    logger.info(f"Louvain metrics - Modularity: {louvain_metrics['modularity']:.4f}, Conductance: {louvain_metrics['avg_conductance']:.4f}")
    logger.info(f"Improvement from baseline - Modularity: +{louvain_metrics['improvement_from_baseline']['modularity']:.4f}")
    
    # Step 3: Refine large communities using Girvan-Newman
    gn_start_time = time.time()
    refined_partition = refine_girvan_newman(G, communities, size_threshold, target_subcommunities)
    
    # Update communities mapping after refinement
    refined_communities = defaultdict(list)
    for node, comm_id in refined_partition.items():
        refined_communities[comm_id].append(node)
    
    gn_metrics = evaluate_all(G, refined_partition, ground_truth)
    gn_time = time.time() - gn_start_time
    
    gn_metrics['runtime'] = gn_time
    gn_metrics['improvement_from_louvain'] = {
        'modularity': gn_metrics['modularity'] - louvain_metrics['modularity'],
        'conductance': louvain_metrics['avg_conductance'] - gn_metrics['avg_conductance']
    }
    all_metrics['girvan_newman'] = gn_metrics
    
    logger.info(f"After refinement: {len(refined_communities)} communities")
    logger.info(f"GN metrics - Modularity: {gn_metrics['modularity']:.4f}, Conductance: {gn_metrics['avg_conductance']:.4f}")
    logger.info(f"Improvement from Louvain - Modularity: {gn_metrics['improvement_from_louvain']['modularity']:.4f}")
    
    # Step 4: Enhance low-modularity communities using Infomap
    infomap_start_time = time.time()
    final_partition = enhance_infomap(G, refined_partition, refined_communities, modularity_threshold)
    
    # Update communities mapping for final partition
    final_communities = defaultdict(list)
    for node, comm_id in final_partition.items():
        final_communities[comm_id].append(node)
    
    infomap_metrics = evaluate_all(G, final_partition, ground_truth)
    infomap_time = time.time() - infomap_start_time
    
    infomap_metrics['runtime'] = infomap_time
    infomap_metrics['improvement_from_gn'] = {
        'modularity': infomap_metrics['modularity'] - gn_metrics['modularity'],
        'conductance': gn_metrics['avg_conductance'] - infomap_metrics['avg_conductance']
    }
    all_metrics['infomap'] = infomap_metrics
    
    logger.info(f"Final communities: {len(final_communities)} communities")
    logger.info(f"Infomap metrics - Modularity: {infomap_metrics['modularity']:.4f}, Conductance: {infomap_metrics['avg_conductance']:.4f}")
    logger.info(f"Improvement from GN - Modularity: {infomap_metrics['improvement_from_gn']['modularity']:.4f}")
    
    # Calculate overall improvements and summary stats
    all_metrics['summary'] = {
        'total_runtime': time.time() - pipeline_start_time,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'final_communities': len(final_communities),
        'total_improvement': {
            'modularity': infomap_metrics['modularity'] - baseline_metrics['modularity'],
            'conductance': baseline_metrics['avg_conductance'] - infomap_metrics['avg_conductance']
        }
    }
    
    # Step 5: Visualize communities
    logger.info("Creating community visualization...")
    visualization_path = plot_communities(G, final_communities)
    
    # Save metrics to file
    metrics_path = save_metrics_to_file(all_metrics)
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    # Generate metrics comparison plot
    plot_path = plot_metrics_comparison(all_metrics)
    logger.info(f"Metrics comparison plot saved to {plot_path}")
    
    # Add NMI results to your summary
    if ground_truth:
        logger.info("=" * 50)
        logger.info("NORMALIZED MUTUAL INFORMATION (NMI) RESULTS")
        logger.info("=" * 50)
        logger.info(f"NMI measures similarity between detected communities and ground truth (0-1, higher is better)")
        logger.info(f"  - Baseline: {baseline_metrics['nmi']:.4f}")
        logger.info(f"  - Louvain:  {louvain_metrics['nmi']:.4f}")
        logger.info(f"  - GN:       {gn_metrics['nmi']:.4f}")
        logger.info(f"  - Infomap:  {infomap_metrics['nmi']:.4f}")
        
        # Add NMI to metrics dictionary
        all_metrics['baseline']['nmi'] = baseline_metrics['nmi']
        all_metrics['louvain']['nmi'] = louvain_metrics['nmi']
        all_metrics['girvan_newman']['nmi'] = gn_metrics['nmi']
        all_metrics['infomap']['nmi'] = infomap_metrics['nmi']
        
        # Add summary of NMI improvements
        all_metrics['summary']['nmi_improvements'] = {
            'louvain_vs_baseline': louvain_metrics['nmi'] - baseline_metrics['nmi'],
            'gn_vs_louvain': gn_metrics['nmi'] - louvain_metrics['nmi'],
            'infomap_vs_gn': infomap_metrics['nmi'] - gn_metrics['nmi'],
            'overall': infomap_metrics['nmi'] - baseline_metrics['nmi']
        }
    
    # Print summary results
    total_runtime = time.time() - pipeline_start_time
    logger.info("=" * 50)
    logger.info("COMMUNITY DETECTION PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Nodes: {G.number_of_nodes()}")
    logger.info(f"Edges: {G.number_of_edges()}")
    logger.info(f"Final number of communities: {len(final_communities)}")
    logger.info(f"Stage-by-stage modularity:")
    logger.info(f"  - Baseline: {baseline_metrics['modularity']:.4f}")
    logger.info(f"  - Louvain:  {louvain_metrics['modularity']:.4f} (delta: +{louvain_metrics['improvement_from_baseline']['modularity']:.4f})")
    logger.info(f"  - GN:       {gn_metrics['modularity']:.4f} (delta: {gn_metrics['improvement_from_louvain']['modularity']:.4f})")
    logger.info(f"  - Infomap:  {infomap_metrics['modularity']:.4f} (delta: {infomap_metrics['improvement_from_gn']['modularity']:.4f})")
    logger.info(f"Overall improvement: +{all_metrics['summary']['total_improvement']['modularity']:.4f}")
    logger.info(f"Visualization saved to: {visualization_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    logger.info("=" * 50)
    
if __name__ == "__main__":
    main()

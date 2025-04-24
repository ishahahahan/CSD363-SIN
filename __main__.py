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
import networkx as nx

# Import pipeline modules
from data_io import get_graph
from detection import run_louvain, refine_girvan_newman, enhance_infomap
from evaluation import evaluate_all, compare_algorithms, track_algorithm_metrics
from visualization import plot_communities
from visualization.metrics_visualization import (
    plot_algorithm_metrics,
    plot_edge_betweenness_distribution,
    plot_description_length_contribution
)

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
    
    # Output log file handler (INFO level - same as console output)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_log_path = os.path.join("results", f"output_{timestamp}.txt")
    
    # Make sure the results directory exists
    os.makedirs("results", exist_ok=True)
    
    output_handler = logging.FileHandler(output_log_path)
    output_handler.setLevel(logging.INFO)
    output_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    output_handler.setFormatter(output_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(output_handler)
    
    logger.info(f"Output log being saved to: {output_log_path}")
    
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
    
    # Data settings
    parser.add_argument('--data-dir', default='data',
                        help='Directory to store/find data (default: data)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for graph (default: None, use full graph)')
    
    # Algorithm parameters
    parser.add_argument('--size-threshold', type=int, default=1000,
                        help='Size threshold for community refinement (default: 1000)')
    parser.add_argument('--target-subcommunities', type=int, default=5,
                        help='Target number of subcommunities for refinement (default: 5)')
    parser.add_argument('--modularity-threshold', type=float, default=0.3,
                        help='Modularity threshold for Infomap enhancement (default: 0.3)')
    
    # Performance parameters
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum iterations for Girvan-Newman algorithm')
    parser.add_argument('--time-limit', type=int, default=600,
                        help='Time limit in seconds for each algorithm stage (default: 600)')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Enable fast mode for large graphs (uses approximations)')
                        
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (YAML or JSON)')
    parser.add_argument('--input-edge-file', type=str, default=None,
                        help='Path to input edge file for graph construction')
    parser.add_argument('--ground-truth-file', type=str, default=None,
                        help='Path to ground truth file for evaluation')
    
    return parser.parse_args()

def make_json_serializable(obj):
    """
    Recursively converts non-JSON-serializable objects to serializable formats.
    
    Args:
        obj: Any Python object
        
    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        # Convert dictionary with potentially non-string keys
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Convert elements of a list
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuple to string
        return str(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        # These types are already JSON-serializable
        return obj
    else:
        # Convert anything else to string
        return str(obj)

def save_metrics_to_file(all_metrics, output_dir='results'):
    """
    Save metrics dictionary to a JSON file with proper error handling.
    
    Args:
        all_metrics: Metrics dictionary to save
        output_dir: Directory to save the file
        
    Returns:
        str: Path to saved file or None if error occurred
    """
    logger = logging.getLogger('community_pipeline')
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp-based filename to ensure uniqueness
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(output_dir, f"metrics_{timestamp}.json")
        
        # Convert non-serializable objects to serializable format
        serializable_metrics = make_json_serializable(all_metrics)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        logger.info(f"Evaluation metrics saved to {filepath}")
        return filepath
    
    except OSError as e:
        logger.error(f"Error saving metrics file: {str(e)}")
        # Try with a simpler filename in the current directory
        try:
            simple_filepath = f"metrics_{timestamp}.json"
            with open(simple_filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Evaluation metrics saved to {simple_filepath} (fallback)")
            return simple_filepath
        except Exception as e2:
            logger.error(f"Failed to save metrics even with simple filename: {str(e2)}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error saving metrics file: {str(e)}")
        return None

def analyze_graph_structure(G):
    """
    Analyze the graph structure to provide context for evaluation metrics.
    
    Args:
        G (networkx.Graph): Input graph
        
    Returns:
        dict: Dictionary of graph structure metrics
    """
    logger = logging.getLogger('community_pipeline')
    logger.info("Analyzing graph structure...")
    
    # Basic graph properties
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Connectivity analysis
    connected_components = list(nx.connected_components(G))
    n_components = len(connected_components)
    largest_cc_size = len(max(connected_components, key=len))
    component_sizes = sorted([len(cc) for cc in connected_components], reverse=True)
    
    # Degree statistics
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    
    # Create analysis report
    analysis = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "connected_components": n_components,
        "largest_component_size": largest_cc_size,
        "largest_component_percentage": largest_cc_size / n_nodes * 100 if n_nodes > 0 else 0,
        "component_sizes": component_sizes[:10],  # Show only top 10 components
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "isolated_nodes": sum(1 for d in degrees if d == 0)
    }
    
    # Log important insights
    logger.info(f"Graph has {n_nodes} nodes and {n_edges} edges (density: {density:.6f})")
    logger.info(f"Graph has {n_components} connected components")
    logger.info(f"Largest component has {largest_cc_size} nodes ({analysis['largest_component_percentage']:.2f}% of graph)")
    
    if n_components > 1:
        logger.warning(f"Graph is disconnected with {n_components} components - metrics may be affected")
        component_distribution = [f"{size} nodes: {component_sizes.count(size)} components" 
                                 for size in sorted(set(component_sizes[:10]), reverse=True)]
        logger.info(f"Component size distribution: {', '.join(component_distribution)}")
    
    if analysis['isolated_nodes'] > 0:
        logger.warning(f"Graph contains {analysis['isolated_nodes']} isolated nodes")
    
    return analysis

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

def load_edge_file_ground_truth(file_path):
    """
    Load ground truth from the edge file, which could be in one of these formats:
    1. node_id community_id (one per line)
    2. Edge list format that needs to be converted to communities
    
    Args:
        file_path (str): Path to the edge file
        
    Returns:
        dict: Dictionary mapping node_id to community_id
    """
    logger = logging.getLogger('community_pipeline')
    ground_truth = {}
    
    try:
        logger.info(f"Attempting to load ground truth from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Edge file does not exist: {file_path}")
            return {}
            
        # First, try to read as node_id community_id format
        node_to_comm_format = True
        edge_list = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip() and not line.startswith('#'):  # Skip comments and empty lines
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # Try to interpret as node_id community_id
                            node_id = int(parts[0])
                            comm_id = int(parts[1])
                            ground_truth[node_id] = comm_id
                        except ValueError:
                            # If conversion fails, it might be edge list format
                            node_to_comm_format = False
                            try:
                                u, v = int(parts[0]), int(parts[1])
                                edge_list.append((u, v))
                            except ValueError:
                                logger.warning(f"Skipping invalid line {line_num} in edge file: {line.strip()}")
                    else:
                        logger.warning(f"Line {line_num} doesn't have enough columns: {line.strip()}")
        
        # If it wasn't in node_id community_id format, try to convert edge list to communities
        if not ground_truth and edge_list:
            logger.info(f"Edge file appears to be in edge list format with {len(edge_list)} edges. Converting to communities...")
            # Create a graph from the edge list
            import networkx as nx
            G = nx.Graph()
            G.add_edges_from(edge_list)
            
            # Get connected components as communities
            for comm_id, component in enumerate(nx.connected_components(G)):
                for node in component:
                    ground_truth[node] = comm_id
            
            logger.info(f"Extracted {len(set(ground_truth.values()))} communities from edge file's connected components")
        
        if ground_truth:
            logger.info(f"Loaded ground truth from edge file with {len(ground_truth)} nodes across {len(set(ground_truth.values()))} communities")
        else:
            logger.warning(f"Could not extract ground truth from edge file: {file_path}")
            
        return ground_truth
        
    except Exception as e:
        logger.error(f"Error loading ground truth from edge file: {str(e)}")
        return {}

def load_ground_truth(data_dir, sample_size=None, edge_file=None, ground_truth_file=None):
    """Load ground truth with support for different formats"""
    logger = logging.getLogger('community_pipeline')
    
    logger.info(f"Looking for ground truth files in: {data_dir}")
    
    # First check if ground_truth_file is specified (this is the new dedicated parameter)
    if ground_truth_file:
        ground_truth_path = os.path.join(data_dir, ground_truth_file)
        logger.info(f"Checking for ground truth file at: {ground_truth_path}")
        
        if os.path.exists(ground_truth_path):
            logger.info(f"Found ground truth file: {ground_truth_path}")
            return load_edge_file_ground_truth(ground_truth_path)
        else:
            logger.warning(f"Specified ground truth file not found: {ground_truth_path}")
    
    # Fall back to edge_file if no ground_truth_file was specified or found
    if edge_file:
        logger.info(f"Falling back to edge file for ground truth: {edge_file}")
        edge_file_path = os.path.join(data_dir, edge_file)
        
        if os.path.exists(edge_file_path):
            logger.info(f"Found edge file: {edge_file_path}")
            return load_edge_file_ground_truth(edge_file_path)
        else:
            logger.warning(f"Specified edge file not found: {edge_file_path}")
    
    # Check for LiveJournal ground truth
    lj_ground_truth_file = os.path.join(data_dir, 'com-lj.top5000.cmty.txt')
    if os.path.exists(lj_ground_truth_file):
        logger.info(f"Loading LiveJournal ground truth from {lj_ground_truth_file}")
        return load_livejournal_ground_truth(lj_ground_truth_file)
    
    logger.warning("No ground truth files found. NMI will not be calculated.")
    return None

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

def plot_metrics_comparison(metrics_dict, output_dir='results', filename='metrics_comparison.png'):
    """Generate a comparison plot of key metrics across pipeline stages"""
    # Get algorithm stages only, filtering out metadata and summary
    algorithm_stages = [k for k in metrics_dict.keys() 
                       if k not in ['summary', 'graph_analysis', 'comparison']]
    
    if not algorithm_stages:
        logger = logging.getLogger('community_pipeline')
        logger.warning("No algorithm stages to plot")
        return None
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the output path
        output_path = os.path.join(output_dir, filename)
        
        # Determine which metrics are available across all stages
        # Standard metrics
        available_metrics = []
        if all('modularity' in metrics_dict[stage] for stage in algorithm_stages):
            available_metrics.append('modularity')
        if all('avg_conductance' in metrics_dict[stage] for stage in algorithm_stages):
            available_metrics.append('avg_conductance')
        if all('num_communities' in metrics_dict[stage] for stage in algorithm_stages):
            available_metrics.append('num_communities')
        if all('coverage' in metrics_dict[stage] for stage in algorithm_stages):
            available_metrics.append('coverage')
        
        # Check if any stage has algorithm-specific metrics
        has_edge_betweenness = any('avg_edge_betweenness' in metrics_dict[stage] for stage in algorithm_stages)
        has_description_length = any('description_length' in metrics_dict[stage] for stage in algorithm_stages)
        
        # Include algorithm-specific metrics
        if has_edge_betweenness:
            available_metrics.append('edge_betweenness')
        if has_description_length:
            available_metrics.append('description_length')
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4*len(available_metrics)))
        
        # Handle case with only one metric
        if len(available_metrics) == 1:
            axes = [axes]
            
        # Plot each available metric
        for i, metric in enumerate(available_metrics):
            if metric == 'edge_betweenness':
                # For edge_betweenness, only plot for stages that have it
                x_values = []
                y_values = []
                for idx, stage in enumerate(algorithm_stages):
                    if 'avg_edge_betweenness' in metrics_dict[stage]:
                        x_values.append(idx)
                        y_values.append(metrics_dict[stage]['avg_edge_betweenness'])
                
                if x_values:  # Only plot if we have data
                    axes[i].plot(x_values, y_values, 'o-', linewidth=2, markersize=8, color='purple')
                    axes[i].set_title('Average Edge Betweenness')
                    axes[i].set_xticks(range(len(algorithm_stages)))
                    axes[i].set_xticklabels(algorithm_stages, rotation=45, ha='right')
                    axes[i].set_ylabel('Edge Betweenness')
                    axes[i].grid(True)
            
            elif metric == 'description_length':
                # For description_length, only plot for stages that have it
                x_values = []
                y_values = []
                for idx, stage in enumerate(algorithm_stages):
                    if 'description_length' in metrics_dict[stage]:
                        x_values.append(idx)
                        y_values.append(metrics_dict[stage]['description_length'])
                
                if x_values:  # Only plot if we have data
                    axes[i].plot(x_values, y_values, 'o-', linewidth=2, markersize=8, color='brown')
                    axes[i].set_title('Description Length')
                    axes[i].set_xticks(range(len(algorithm_stages)))
                    axes[i].set_xticklabels(algorithm_stages, rotation=45, ha='right')
                    axes[i].set_ylabel('Description Length')
                    axes[i].grid(True)
            
            else:
                # For standard metrics available in all stages
                metric_key = 'avg_conductance' if metric == 'avg_conductance' else metric
                metric_values = [metrics_dict[stage].get(metric_key, 0) for stage in algorithm_stages]
                axes[i].plot(range(len(algorithm_stages)), metric_values, 'o-', linewidth=2, markersize=8)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xticks(range(len(algorithm_stages)))
                axes[i].set_xticklabels(algorithm_stages, rotation=45, ha='right')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].grid(True)
        
        # Add note about disconnected graph if needed
        if 'graph_analysis' in metrics_dict and metrics_dict['graph_analysis']['connected_components'] > 1:
            components = metrics_dict['graph_analysis']['connected_components']
            fig.text(0.5, 0.01, f"Note: Graph has {components} connected components",
                    ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger = logging.getLogger('community_pipeline')
        logger.info(f"Metrics comparison plot saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger = logging.getLogger('community_pipeline')
        logger.error(f"Error generating metrics comparison plot: {e}")
        return None

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
    
    # Performance parameters
    max_iterations = args.max_iterations if hasattr(args, 'max_iterations') and args.max_iterations is not None else config.get('max_iterations', None)
    time_limit = args.time_limit if hasattr(args, 'time_limit') and args.time_limit != 600 else config.get('time_limit', 600)
    fast_mode = args.fast_mode or config.get('fast_mode', False)
    
    # Get file paths from config
    input_edge_file = args.input_edge_file if hasattr(args, 'input_edge_file') and args.input_edge_file is not None else config.get('input_edge_file', None)
    ground_truth_file = args.ground_truth_file if hasattr(args, 'ground_truth_file') and args.ground_truth_file is not None else config.get('ground_truth_file', None)
    
    logger.info(f"Configuration: data_dir={data_dir}, sample_size={sample_size}, "
               f"size_threshold={size_threshold}, target_subcommunities={target_subcommunities}, "
               f"modularity_threshold={modularity_threshold}")
    logger.info(f"Performance settings: max_iterations={max_iterations}, time_limit={time_limit}s, fast_mode={fast_mode}")
    logger.info(f"File settings: input_edge_file={input_edge_file}, ground_truth_file={ground_truth_file}")
    
    # Dictionary to store metrics at each pipeline stage
    all_metrics = {}
    
    # Create results directory at the beginning
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load the graph
    load_start_time = time.time()
    url = "https://snap.stanford.edu/data/com-LiveJournal.tar.gz"
    
    # Load the graph from the input edge file
    G = get_graph(data_dir, edge_file_path=input_edge_file, url=url, sample_size=sample_size)
    load_time = time.time() - load_start_time
    logger.info(f"Graph loaded in {load_time:.2f} seconds")
    
    # If input_edge_file is specified but doesn't exist, create it from the loaded graph
    if input_edge_file:
        from data_io import ensure_ground_truth_file
        ensure_ground_truth_file(G, data_dir, input_edge_file)

    if ground_truth_file and ground_truth_file != input_edge_file:
        from data_io import ensure_ground_truth_file
        ensure_ground_truth_file(G, data_dir, ground_truth_file)
    
    # Add graph structure analysis for context
    graph_analysis = analyze_graph_structure(G)
    
    # Dictionary to store metrics at each pipeline stage
    all_metrics = {
        'graph_analysis': graph_analysis
    }
    
    # Load ground truth (add this after loading the graph)
    ground_truth = None
    if config.get('use_ground_truth', True):
        # Make sure we're using the correct ground truth file
        if ground_truth_file:
            logger.info(f"Using ground truth file: {ground_truth_file}")
            
        # Try to load ground truth from specified ground truth file
        ground_truth = load_ground_truth(data_dir, sample_size=sample_size, edge_file=input_edge_file, ground_truth_file=ground_truth_file)
        
        # If using a sampled graph, filter the ground truth
        if sample_size and ground_truth:
            ground_truth = filter_ground_truth_for_sample(ground_truth, G)
        
        # If no ground truth available but testing is needed, create synthetic
        if not ground_truth and config.get('use_synthetic_ground_truth', False):
            synthetic_method = config.get('synthetic_ground_truth_method', 'louvain')
            ground_truth = create_synthetic_ground_truth(G, method=synthetic_method)
            logger.info(f"Created synthetic ground truth with {len(ground_truth)} nodes")
        
        if ground_truth:
            logger.info(f"Ground truth loaded successfully with {len(ground_truth)} nodes and {len(set(ground_truth.values()))} communities")
        else:
            # If still no ground truth, create it from components as a last resort
            if config.get('generate_ground_truth_from_components', True):
                logger.warning("No ground truth loaded. Generating ground truth from graph components...")
                ground_truth = {}
                for i, component in enumerate(nx.connected_components(G)):
                    for node in component:
                        ground_truth[node] = i
                logger.info(f"Generated ground truth from {len(set(ground_truth.values()))} connected components")
    
    # BASELINE EVALUATION - Single Community
    baseline_start_time = time.time()
    baseline_partition = {node: 0 for node in G.nodes()}
    baseline_metrics = evaluate_all(G, baseline_partition, ground_truth, algorithm_type='baseline')
    baseline_time = time.time() - baseline_start_time
    
    baseline_metrics['runtime'] = baseline_time
    all_metrics['baseline'] = baseline_metrics
    logger.info(f"Baseline metrics - Modularity: {baseline_metrics['modularity']:.4f}, Conductance: {baseline_metrics['avg_conductance']:.4f}")
    
    # Step 2: Run initial community detection using Louvain
    louvain_start_time = time.time()
    partition, communities = run_louvain(G)
    louvain_metrics = evaluate_all(G, partition, ground_truth, algorithm_type='louvain')
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
    
    # For large graphs in fast mode or with many components, skip GN refinement if not needed
    skip_gn = False
    if (fast_mode and G.number_of_nodes() > 50000) or graph_analysis['connected_components'] > 5000:
        logger.info("Large graph with many components: checking if Girvan-Newman refinement is needed")
        # Check if we already have good communities from Louvain
        if louvain_metrics['modularity'] > 0.9:
            logger.info("High modularity detected from Louvain. Skipping Girvan-Newman refinement.")
            skip_gn = True
            refined_partition = partition
            refined_communities = communities
            
            # Copy metrics to maintain pipeline
            gn_metrics = louvain_metrics.copy()
            gn_metrics['skipped'] = True
            gn_metrics['runtime'] = 0.0
            gn_metrics['improvement_from_louvain'] = {
                'modularity': 0.0,
                'conductance': 0.0
            }
    
    if not skip_gn:
        # Set max_iterations for GN algorithm
        if max_iterations is None:
            # Adjust max_iterations based on graph size
            if G.number_of_nodes() > 50000:
                max_iterations = 20
            elif G.number_of_nodes() > 10000:
                max_iterations = 50
            else:
                max_iterations = 100
        
        logger.info(f"Running Girvan-Newman with max_iterations={max_iterations}")
        refined_partition = refine_girvan_newman(G, communities, size_threshold, target_subcommunities, 
                                                max_iterations=max_iterations)
        
        # Update communities mapping after refinement
        refined_communities = defaultdict(list)
        for node, comm_id in refined_partition.items():
            refined_communities[comm_id].append(node)
        
        gn_metrics = evaluate_all(G, refined_partition, ground_truth, algorithm_type='girvan_newman')
        gn_time = time.time() - gn_start_time
        
        gn_metrics['runtime'] = gn_time
        gn_metrics['improvement_from_louvain'] = {
            'modularity': gn_metrics['modularity'] - louvain_metrics['modularity'],
            'conductance': louvain_metrics['avg_conductance'] - gn_metrics['avg_conductance']
        }
    
    all_metrics['girvan_newman'] = gn_metrics
    
    logger.info(f"After refinement: {len(refined_communities)} communities")
    logger.info(f"GN metrics - Modularity: {gn_metrics['modularity']:.4f}, Conductance: {gn_metrics['avg_conductance']:.4f}")
    
    # Step 4: Enhance low-modularity communities using Infomap
    infomap_start_time = time.time()
    final_partition = enhance_infomap(G, refined_partition, refined_communities, modularity_threshold)
    
    # Update communities mapping for final partition
    final_communities = defaultdict(list)
    for node, comm_id in final_partition.items():
        final_communities[comm_id].append(node)
    
    infomap_metrics = evaluate_all(G, final_partition, ground_truth, algorithm_type='infomap')
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
        },
        'graph_structure': {
            'n_components': graph_analysis['connected_components'],
            'largest_component_pct': graph_analysis['largest_component_percentage'],
            'density': graph_analysis['density'],
            'avg_degree': graph_analysis['avg_degree'],
        }
    }
    
    # Step 5: Visualize communities
    logger.info("Creating community visualization...")
    visualization_path = plot_communities(G, final_communities)
    
    # Save metrics to file
    metrics_path = save_metrics_to_file(all_metrics, output_dir=results_dir)
    if metrics_path:
        logger.info(f"Evaluation metrics saved to {metrics_path}")
    else:
        logger.warning("Failed to save metrics to file")
    
    # Generate metrics comparison plot
    plot_path = plot_metrics_comparison(all_metrics, output_dir=results_dir)
    if plot_path:
        logger.info(f"Metrics comparison plot saved to {plot_path}")
    else:
        logger.warning("Could not generate metrics comparison plot")
    
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
    
    # Print enhanced summary results with context for metrics
    total_runtime = time.time() - pipeline_start_time
    logger.info("=" * 50)
    logger.info("COMMUNITY DETECTION PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    logger.info(f"Graph density: {graph_analysis['density']:.6f}")
    logger.info(f"Connected components: {graph_analysis['connected_components']}")
    
    # If graph is highly disconnected, add warning
    if graph_analysis['connected_components'] > G.number_of_nodes() * 0.1:  # More than 10% of nodes are disconnected components
        logger.warning("IMPORTANT: Graph is highly disconnected - metrics may not be meaningful")
        logger.warning(f"High number of connected components ({graph_analysis['connected_components']}) may explain high modularity values")
    
    logger.info(f"Final number of communities: {len(final_communities)}")
    logger.info(f"Stage-by-stage modularity:")
    logger.info(f"  - Baseline: {baseline_metrics['modularity']:.4f}")
    logger.info(f"  - Louvain:  {louvain_metrics['modularity']:.4f} (delta: +{louvain_metrics['improvement_from_baseline']['modularity']:.4f})")
    logger.info(f"  - GN:       {gn_metrics['modularity']:.4f} (delta: {gn_metrics['improvement_from_louvain']['modularity']:.4f})")
    logger.info(f"  - Infomap:  {infomap_metrics['modularity']:.4f} (delta: {infomap_metrics['improvement_from_gn']['modularity']:.4f})")
    logger.info(f"Overall improvement: +{all_metrics['summary']['total_improvement']['modularity']:.4f}")
    
    # Add interpretation of the results based on graph structure
    if graph_analysis['connected_components'] > 100 and louvain_metrics['modularity'] > 0.9:
        logger.info("NOTE: The high modularity values are likely due to many disconnected components in the graph")
        logger.info("      Each component forms its own natural community with no external edges")
    
    logger.info(f"Visualization saved to: {visualization_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")
    logger.info("=" * 50)
    
    # Add log completion message to make sure output log is properly saved
    logger.info("Pipeline execution completed successfully.")
    logger.info(f"Output log has been saved to: {os.path.join('results', f'output_{time.strftime("%Y%m%d-%H%M%S")}.txt')}")
    
    # Return some key results for testing/verification
    return {
        'modularity': infomap_metrics['modularity'],
        'communities': len(final_communities),
        'visualization_path': visualization_path,
        'metrics_path': metrics_path
    }
    
if __name__ == "__main__":
    try:
        result = main()
        # Code execution completed, no need for additional output here
    except Exception as e:
        logger = logging.getLogger('community_pipeline')
        logger.error(f"Pipeline execution failed with error: {str(e)}")
        # Print stack trace to help with debugging
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

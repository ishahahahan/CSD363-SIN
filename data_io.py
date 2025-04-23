import os
import urllib.request
import tarfile
import pickle
import logging
import networkx as nx
from tqdm import tqdm

logger = logging.getLogger(__name__)

def use_existing_file(edge_file_path, data_dir):
    """
    Use an existing edge list file instead of downloading.
    
    Args:
        edge_file_path (str): Path to the existing edge list file
        data_dir (str): Directory to store processed files
    
    Returns:
        str: Path to the edge list file
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(edge_file_path):
        raise FileNotFoundError(f"Edge file not found at: {edge_file_path}")
    
    logger.info(f"Using existing edge file at {edge_file_path}")
    return edge_file_path

def download_and_extract(url, data_dir):
    """
    Download and extract the LiveJournal dataset from the given URL.
    
    Args:
        url (str): URL to download the dataset from
        data_dir (str): Directory to store downloaded and extracted files
    
    Returns:
        str: Path to the extracted edge list file
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # File paths
    tar_path = os.path.join(data_dir, "com-LiveJournal.tar.gz")
    edge_file = os.path.join(data_dir, "com-lj.ungraph.txt")
    
    # Download if file doesn't exist
    if not os.path.exists(tar_path):
        logger.info(f"Downloading LiveJournal dataset from {url}")
        urllib.request.urlretrieve(url, tar_path)
        logger.info(f"Dataset downloaded to {tar_path}")
    else:
        logger.info(f"Using existing download at {tar_path}")
    
    # Extract if edge file doesn't exist
    if not os.path.exists(edge_file):
        logger.info(f"Extracting dataset from {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            # Only extract the edge list file
            for member in tar.getmembers():
                if member.name.endswith("com-lj.ungraph.txt"):
                    member.name = os.path.basename(member.name)  # Remove directory structure
                    tar.extract(member, path=data_dir)
                    logger.info(f"Extracted edge file to {edge_file}")
                    break
    else:
        logger.info(f"Using existing edge file at {edge_file}")
    
    return edge_file

def load_graph(edge_file, sample_size=None):
    """
    Load graph from edge list file, with optional sampling.
    
    Args:
        edge_file (str): Path to the edge list file
        sample_size (int, optional): Number of edges to sample (None for all)
    
    Returns:
        networkx.Graph: The loaded graph
    """
    logger.info(f"Loading graph from {edge_file}")
    
    # Create an empty graph
    G = nx.Graph()
    
    # Read the edge list with sampling if needed
    if sample_size:
        logger.info(f"Sampling first {sample_size} edges")
        edge_count = 0
        with open(edge_file, 'r') as f:
            for line in tqdm(f):
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Split the line and try to parse as integers
                parts = line.strip().split()
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line: {line.strip()}")
                    continue
                
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                    edge_count += 1
                    
                    if edge_count >= sample_size:
                        break
                except ValueError:
                    logger.warning(f"Skipping line with non-integer node IDs: {line.strip()}")
    else:
        logger.info("Loading full graph (this may take a while)")
        with open(edge_file, 'r') as f:
            for line in tqdm(f):
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Split the line and try to parse as integers
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                except ValueError:
                    logger.warning(f"Skipping line with non-integer node IDs: {line.strip()}")
    
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def save_graph_as_edgelist(G, filepath):
    """
    Save a graph as an edge list file.
    
    Args:
        G (networkx.Graph): Graph to save
        filepath (str): Path where the edge list will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger('community_pipeline')
    try:
        logger.info(f"Saving graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to {filepath}")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save as edge list
        with open(filepath, 'w') as f:
            f.write("# Edge list format: node_id node_id\n")
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
                
        logger.info(f"Graph successfully saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving graph to edge list: {str(e)}")
        return False

def ensure_ground_truth_file(G, data_dir, edge_file_path):
    """
    Ensure that the ground truth file exists, creating it from the graph if needed.
    
    Args:
        G (networkx.Graph): Input graph
        data_dir (str): Directory to store data
        edge_file_path (str): Path to edge file (relative to data_dir)
        
    Returns:
        str: Path to the ground truth file
    """
    logger = logging.getLogger('community_pipeline')
    
    full_path = os.path.join(data_dir, edge_file_path)
    
    if not os.path.exists(full_path):
        logger.info(f"Ground truth file {full_path} not found, creating from graph")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            success = save_graph_as_edgelist(G, full_path)
            if not success:
                logger.error(f"Failed to create ground truth file {full_path}")
                return None
        except Exception as e:
            logger.error(f"Error creating ground truth file: {str(e)}")
            return None
    
    logger.info(f"Ground truth file is available at: {full_path}")
    return full_path

def get_graph(data_dir, edge_file_path=None, url="https://snap.stanford.edu/data/com-LiveJournal.tar.gz", sample_size=None, force_reload=False):
    """
    Get the graph, using cached pickle if available or a pre-downloaded edge file.
    
    Args:
        data_dir (str): Directory to store/find data
        edge_file_path (str, optional): Path to pre-downloaded edge file
        url (str): URL to download from if needed and no edge file is provided
        sample_size (int, optional): Number of edges to sample
        force_reload (bool): Whether to force reload even if pickle exists
    
    Returns:
        networkx.Graph: The loaded graph
    """
    logger = logging.getLogger('community_pipeline')
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Define pickle filename based on sample size
    pickle_suffix = f"_{sample_size}" if sample_size else ""
    pickle_path = os.path.join(data_dir, f"livejournal{pickle_suffix}.pkl")
    
    # Check if edge_file_path is specified for generating ground truth
    if edge_file_path:
        edge_file_full_path = os.path.join(data_dir, edge_file_path)
        force_reload = not os.path.exists(edge_file_full_path) or force_reload
        
        # If forcing reload but edge file doesn't exist, create it from cached graph
        if force_reload and os.path.exists(pickle_path) and not os.path.exists(edge_file_full_path):
            logger.info(f"Creating edge file {edge_file_full_path} from cached graph")
            with open(pickle_path, 'rb') as f:
                cached_G = pickle.load(f)
                save_graph_as_edgelist(cached_G, edge_file_full_path)
    
    # Load from pickle if exists and not forcing reload
    if os.path.exists(pickle_path) and not force_reload:
        logger.info(f"Loading graph from cached pickle: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # Use specified edge file if provided
    if edge_file_path:
        full_path = os.path.join(data_dir, edge_file_path)
        if os.path.exists(full_path):
            logger.info(f"Loading graph from specified edge file: {full_path}")
            G = load_graph(full_path, sample_size)
            
            if G:
                logger.info(f"Successfully loaded graph from {full_path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                # Cache the graph
                logger.info(f"Saving graph to pickle: {pickle_path}")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(G, f)
                return G
        else:
            logger.warning(f"Specified edge file not found: {full_path}")
    
    # Use existing edge file if provided, otherwise download and extract
    edge_file = download_and_extract(url, data_dir)
    
    G = load_graph(edge_file, sample_size)
    
    # Cache the graph
    logger.info(f"Saving graph to pickle: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    
    return G

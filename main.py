"""
Main entry point for the Hybrid Community Detection project.
This script runs experiments for community detection on the LiveJournal dataset.
"""

import os
import time
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import json

from data_loader import LiveJournalLoader
from algorithms.louvain_algorithm import detect_communities as louvain_detect
from algorithms.girvan_newman import detect_communities as gn_detect, refine_communities
from algorithms.infomap_algorithm import detect_communities as infomap_detect, handle_boundary_nodes
from algorithms.hybrid_detector import detect_communities as hybrid_detect
from evaluation.metrics import evaluate_all, print_evaluation_summary
from visualization import plot_communities, plot_comparison, save_community_graph
from utils import save_results, load_results, setup_logging, timer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hybrid Community Detection')
    
    parser.add_argument('--sample_size', type=int, default=5000,
                        help='Number of nodes to sample from LiveJournal dataset')
    parser.add_argument('--max_communities', type=int, default=50,
                        help='Maximum number of communities to consider when sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['louvain', 'girvan_newman', 'infomap', 'hybrid', 'all'],
                        help='Which algorithm to run')
    parser.add_argument('--save_graph', action='store_true',
                        help='Save the graph structure to a file')
    parser.add_argument('--use_cached', action='store_true',
                        help='Use cached results if available')
    
    return parser.parse_args()

def run_louvain(graph, args):
    """Run Louvain algorithm."""
    print("\n=== Running Louvain Method ===")
    communities, execution_time = louvain_detect(graph)
    return communities, execution_time

def run_girvan_newman(graph, args):
    """Run Girvan-Newman algorithm."""
    print("\n=== Running Girvan-Newman Algorithm ===")
    communities, execution_time = gn_detect(graph)
    return communities, execution_time

def run_infomap(graph, args):
    """Run Infomap algorithm."""
    print("\n=== Running Infomap Algorithm ===")
    communities, execution_time = infomap_detect(graph)
    return communities, execution_time

def run_hybrid(graph, args):
    """Run hybrid community detection pipeline."""
    print("\n=== Running Hybrid Community Detection ===")
    
    # Step 1: Use Louvain for initial communities
    print("Step 1: Initial partitioning with Louvain")
    louvain_communities, louvain_time = louvain_detect(graph)
    
    # Step 2: Refine using Girvan-Newman within each Louvain community
    print("Step 2: Refining communities with Girvan-Newman")
    refined_communities = refine_communities(graph, louvain_communities)
    
    # Step 3: Handle boundary nodes with Infomap
    print("Step 3: Handling boundary nodes with Infomap")
    final_communities = handle_boundary_nodes(graph, refined_communities)
    
    total_time = louvain_time + time.time() - time.time()  # Add time from other steps if needed
    
    return final_communities, total_time

def main():
    """Main function to run community detection experiments."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging and output directory
    logger = setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if results already exist
    results_file = os.path.join(args.output_dir, f'results_{args.algorithm}_{args.sample_size}.json')
    
    if args.use_cached and os.path.exists(results_file):
        logger.info(f"Loading cached results from {results_file}")
        results = load_results(results_file)
        
        # Extract needed variables from results
        graph = nx.node_link_graph(results['graph'])
        ground_truth = results['ground_truth']
        communities = {
            algo: {int(k): v for k, v in comm.items()} 
            for algo, comm in results['communities'].items()
        }
        execution_times = results['execution_times']
        
        print(f"Loaded cached results with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    else:
        # Load data
        logger.info("Initializing data loader")
        loader = LiveJournalLoader()
        
        # Download data if not already downloaded
        loader.download_data()
        
        # Create a sample network from LiveJournal
        logger.info(f"Creating a sample network with ~{args.sample_size} nodes")
        graph, ground_truth = loader.sample_network(args.sample_size, args.max_communities, args.seed)
        
        # Print graph information
        print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Ground truth available for {len(ground_truth)} nodes ({len(set(ground_truth.values()))} communities)")
        
        # Initialize results storage
        communities = {}
        execution_times = {}
        
        # Run the selected algorithm(s)
        algorithms = []
        if args.algorithm == 'all' or args.algorithm == 'louvain':
            algorithms.append(('louvain', run_louvain))
        if args.algorithm == 'all' or args.algorithm == 'girvan_newman':
            algorithms.append(('girvan_newman', run_girvan_newman))
        if args.algorithm == 'all' or args.algorithm == 'infomap':
            algorithms.append(('infomap', run_infomap))
        if args.algorithm == 'all' or args.algorithm == 'hybrid':
            algorithms.append(('hybrid', run_hybrid))
        
        for algo_name, algo_func in algorithms:
            with timer() as t:
                logger.info(f"Running {algo_name} algorithm")
                communities[algo_name], exec_time = algo_func(graph, args)
                execution_times[algo_name] = exec_time
            
            logger.info(f"{algo_name} completed in {t.elapsed:.2f} seconds")
        
        # Save results
        results = {
            'graph': nx.node_link_data(graph),
            'ground_truth': ground_truth,
            'communities': {algo: {str(k): v for k, v in comm.items()} for algo, comm in communities.items()},
            'execution_times': execution_times
        }
        
        save_results(results, results_file)
        logger.info(f"Saved results to {results_file}")
    
    # Evaluate results
    for algo_name, algo_communities in communities.items():
        print(f"\n==== Evaluation for {algo_name} ====")
        evaluation = evaluate_all(graph, algo_communities, ground_truth)
        print_evaluation_summary(evaluation)
        print(f"Execution time: {execution_times.get(algo_name, 'N/A')} seconds")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations")
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize individual algorithms
        for algo_name, algo_communities in communities.items():
            out_file = os.path.join(vis_dir, f'{algo_name}_communities.png')
            plot_communities(graph, algo_communities, title=f"{algo_name.title()} Communities", 
                            output_file=out_file, max_nodes=500)
        
        # Compare algorithms
        if len(communities) > 1:
            out_file = os.path.join(vis_dir, 'algorithm_comparison.png')
            plot_comparison(communities, execution_times, output_file=out_file)
        
        # Save community graph for external analysis
        if args.save_graph:
            for algo_name, algo_communities in communities.items():
                out_file = os.path.join(vis_dir, f'{algo_name}_community_graph.gexf')
                save_community_graph(graph, algo_communities, output_file=out_file)
    
    logger.info("Experiment completed!")

if __name__ == "__main__":
    main()
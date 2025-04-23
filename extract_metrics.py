import os
import re
import glob
import argparse
import datetime
import logging

def setup_logging():
    """Set up logging for the metrics extractor"""
    logger = logging.getLogger('metrics_extractor')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def find_latest_log_file(results_dir="results"):
    """Find the most recent log file in the results directory"""
    log_pattern = os.path.join(results_dir, "output_*.txt")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        return None
        
    # Get the most recent file
    latest_log = max(log_files, key=os.path.getctime)
    return latest_log

def extract_metrics(log_file):
    """Extract important metrics from a log file"""
    if not os.path.exists(log_file):
        logger.error(f"Log file not found: {log_file}")
        return None
        
    logger.info(f"Extracting metrics from: {log_file}")
    
    metrics = {
        'timestamp': None,
        'graph': {},
        'baseline': {},
        'louvain': {},
        'girvan_newman': {},
        'infomap': {},
        'improvements': {},
        'runtime': None
    }
    
    # Regular expressions for extracting metrics
    timestamp_pattern = r'\[(.*?)\] INFO - Starting Community'
    graph_nodes_pattern = r'Graph has (\d+) nodes and (\d+) edges \(density: ([0-9.e-]+)\)'
    components_pattern = r'Graph has (\d+) connected components'
    largest_component_pattern = r'Largest component has (\d+) nodes \(([0-9.]+)% of graph\)'
    
    baseline_modularity_pattern = r'Baseline metrics - Modularity: ([0-9.]+), Conductance: ([0-9.]+)'
    louvain_communities_pattern = r'Initial Louvain: (\d+) communities'
    louvain_metrics_pattern = r'Louvain metrics - Modularity: ([0-9.]+), Conductance: ([0-9.]+)'
    louvain_improvement_pattern = r'Improvement from baseline - Modularity: \+([0-9.]+)'
    
    gn_communities_pattern = r'After refinement: (\d+) communities'
    gn_metrics_pattern = r'GN metrics - Modularity: ([0-9.]+), Conductance: ([0-9.]+)'
    
    infomap_communities_pattern = r'Final communities: (\d+) communities'
    infomap_metrics_pattern = r'Infomap metrics - Modularity: ([0-9.]+), Conductance: ([0-9.]+)'
    infomap_improvement_pattern = r'Improvement from GN - Modularity: ([0-9.]+)'
    
    nmi_baseline_pattern = r'  - Baseline: ([0-9.]+)'
    nmi_louvain_pattern = r'  - Louvain:  ([0-9.]+)'
    nmi_gn_pattern = r'  - GN:       ([0-9.]+)'
    nmi_infomap_pattern = r'  - Infomap:  ([0-9.]+)'
    
    total_runtime_pattern = r'Total runtime: ([0-9.]+) seconds'
    
    # Read the log file and extract metrics
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract timestamp
        timestamp_match = re.search(timestamp_pattern, content)
        if timestamp_match:
            metrics['timestamp'] = timestamp_match.group(1)
        
        # Extract graph statistics
        graph_match = re.search(graph_nodes_pattern, content)
        if graph_match:
            metrics['graph']['nodes'] = int(graph_match.group(1))
            metrics['graph']['edges'] = int(graph_match.group(2))
            metrics['graph']['density'] = float(graph_match.group(3))
            
        components_match = re.search(components_pattern, content)
        if components_match:
            metrics['graph']['components'] = int(components_match.group(1))
            
        largest_comp_match = re.search(largest_component_pattern, content)
        if largest_comp_match:
            metrics['graph']['largest_component'] = int(largest_comp_match.group(1))
            metrics['graph']['largest_component_pct'] = float(largest_comp_match.group(2))
        
        # Extract baseline metrics
        baseline_match = re.search(baseline_modularity_pattern, content)
        if baseline_match:
            metrics['baseline']['modularity'] = float(baseline_match.group(1))
            metrics['baseline']['conductance'] = float(baseline_match.group(2))
        
        # Extract Louvain metrics
        louvain_comm_match = re.search(louvain_communities_pattern, content)
        if louvain_comm_match:
            metrics['louvain']['communities'] = int(louvain_comm_match.group(1))
            
        louvain_metrics_match = re.search(louvain_metrics_pattern, content)
        if louvain_metrics_match:
            metrics['louvain']['modularity'] = float(louvain_metrics_match.group(1))
            metrics['louvain']['conductance'] = float(louvain_metrics_match.group(2))
            
        louvain_improvement_match = re.search(louvain_improvement_pattern, content)
        if louvain_improvement_match:
            metrics['improvements']['louvain_vs_baseline'] = float(louvain_improvement_match.group(1))
        
        # Extract Girvan-Newman metrics
        gn_comm_match = re.search(gn_communities_pattern, content)
        if gn_comm_match:
            metrics['girvan_newman']['communities'] = int(gn_comm_match.group(1))
            
        gn_metrics_match = re.search(gn_metrics_pattern, content)
        if gn_metrics_match:
            metrics['girvan_newman']['modularity'] = float(gn_metrics_match.group(1))
            metrics['girvan_newman']['conductance'] = float(gn_metrics_match.group(2))
        
        # Extract Infomap metrics
        infomap_comm_match = re.search(infomap_communities_pattern, content)
        if infomap_comm_match:
            metrics['infomap']['communities'] = int(infomap_comm_match.group(1))
            
        infomap_metrics_match = re.search(infomap_metrics_pattern, content)
        if infomap_metrics_match:
            metrics['infomap']['modularity'] = float(infomap_metrics_match.group(1))
            metrics['infomap']['conductance'] = float(infomap_metrics_match.group(2))
            
        infomap_improvement_match = re.search(infomap_improvement_pattern, content)
        if infomap_improvement_match:
            metrics['improvements']['infomap_vs_gn'] = float(infomap_improvement_match.group(1))
        
        # Extract NMI scores
        nmi_baseline_match = re.search(nmi_baseline_pattern, content)
        if nmi_baseline_match:
            metrics['baseline']['nmi'] = float(nmi_baseline_match.group(1))
            
        nmi_louvain_match = re.search(nmi_louvain_pattern, content)
        if nmi_louvain_match:
            metrics['louvain']['nmi'] = float(nmi_louvain_match.group(1))
            
        nmi_gn_match = re.search(nmi_gn_pattern, content)
        if nmi_gn_match:
            metrics['girvan_newman']['nmi'] = float(nmi_gn_match.group(1))
            
        nmi_infomap_match = re.search(nmi_infomap_pattern, content)
        if nmi_infomap_match:
            metrics['infomap']['nmi'] = float(nmi_infomap_match.group(1))
        
        # Extract total runtime
        runtime_match = re.search(total_runtime_pattern, content)
        if runtime_match:
            metrics['runtime'] = float(runtime_match.group(1))
    
    return metrics

def format_metrics_summary(metrics):
    """Format metrics into a readable text summary"""
    if not metrics:
        return "No metrics available."
    
    summary = []
    
    # Add header
    summary.append("=" * 80)
    summary.append("COMMUNITY DETECTION PIPELINE - METRICS SUMMARY")
    if metrics['timestamp']:
        summary.append(f"Run date: {metrics['timestamp']}")
    summary.append("=" * 80)
    
    # Graph statistics
    summary.append("\nGRAPH STATISTICS:")
    summary.append("-" * 50)
    if metrics['graph']:
        summary.append(f"Nodes: {metrics['graph'].get('nodes', 'N/A')}")
        summary.append(f"Edges: {metrics['graph'].get('edges', 'N/A')}")
        summary.append(f"Density: {metrics['graph'].get('density', 'N/A'):.8f}")
        summary.append(f"Connected Components: {metrics['graph'].get('components', 'N/A')}")
        if 'largest_component' in metrics['graph']:
            summary.append(f"Largest Component: {metrics['graph']['largest_component']} nodes " +
                          f"({metrics['graph'].get('largest_component_pct', 0):.2f}% of graph)")
    
    # Algorithm Performance
    summary.append("\nALGORITHM PERFORMANCE:")
    summary.append("-" * 50)
    summary.append("{:<20} {:<15} {:<15} {:<15}".format("Algorithm", "Communities", "Modularity", "Conductance"))
    summary.append("-" * 65)
    
    # Baseline
    if metrics['baseline']:
        summary.append("{:<20} {:<15} {:<15.4f} {:<15.4f}".format(
            "Baseline",
            "1",
            metrics['baseline'].get('modularity', 0),
            metrics['baseline'].get('conductance', 0)
        ))
    
    # Louvain
    if metrics['louvain']:
        summary.append("{:<20} {:<15} {:<15.4f} {:<15.4f}".format(
            "Louvain",
            metrics['louvain'].get('communities', 'N/A'),
            metrics['louvain'].get('modularity', 0),
            metrics['louvain'].get('conductance', 0)
        ))
    
    # Girvan-Newman
    if metrics['girvan_newman']:
        summary.append("{:<20} {:<15} {:<15.4f} {:<15.4f}".format(
            "Girvan-Newman",
            metrics['girvan_newman'].get('communities', 'N/A'),
            metrics['girvan_newman'].get('modularity', 0),
            metrics['girvan_newman'].get('conductance', 0)
        ))
    
    # Infomap
    if metrics['infomap']:
        summary.append("{:<20} {:<15} {:<15.4f} {:<15.4f}".format(
            "Infomap",
            metrics['infomap'].get('communities', 'N/A'),
            metrics['infomap'].get('modularity', 0),
            metrics['infomap'].get('conductance', 0)
        ))
    
    # NMI Comparison
    if any('nmi' in metrics[alg] for alg in ['baseline', 'louvain', 'girvan_newman', 'infomap']):
        summary.append("\nNORMALIZED MUTUAL INFORMATION (NMI):")
        summary.append("-" * 50)
        summary.append("{:<20} {:<15}".format("Algorithm", "NMI Score"))
        summary.append("-" * 35)
        
        if 'nmi' in metrics['baseline']:
            summary.append("{:<20} {:<15.4f}".format("Baseline", metrics['baseline']['nmi']))
        if 'nmi' in metrics['louvain']:
            summary.append("{:<20} {:<15.4f}".format("Louvain", metrics['louvain']['nmi']))
        if 'nmi' in metrics['girvan_newman']:
            summary.append("{:<20} {:<15.4f}".format("Girvan-Newman", metrics['girvan_newman']['nmi']))
        if 'nmi' in metrics['infomap']:
            summary.append("{:<20} {:<15.4f}".format("Infomap", metrics['infomap']['nmi']))
    
    # Improvement Summary
    summary.append("\nIMPROVEMENT SUMMARY:")
    summary.append("-" * 50)
    
    if 'louvain_vs_baseline' in metrics['improvements']:
        summary.append(f"Louvain vs Baseline (Modularity): +{metrics['improvements']['louvain_vs_baseline']:.4f}")
    
    if 'infomap_vs_gn' in metrics['improvements']:
        summary.append(f"Infomap vs Girvan-Newman (Modularity): +{metrics['improvements']['infomap_vs_gn']:.4f}")
    
    # Calculate overall improvement
    if 'modularity' in metrics['baseline'] and 'modularity' in metrics['infomap']:
        overall = metrics['infomap']['modularity'] - metrics['baseline']['modularity']
        summary.append(f"Overall Improvement (Modularity): +{overall:.4f}")
    
    # Runtime
    if metrics['runtime']:
        minutes, seconds = divmod(metrics['runtime'], 60)
        summary.append(f"\nTotal Runtime: {int(minutes)}m {seconds:.2f}s")
    
    return "\n".join(summary)

def save_metrics_summary(metrics, output_file=None):
    """Save formatted metrics to a file"""
    summary = format_metrics_summary(metrics)
    
    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"results/metrics_summary_{timestamp}.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Metrics summary saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Extract and summarize metrics from community detection logs')
    parser.add_argument('--log-file', help='Path to log file (default: most recent in results dir)')
    parser.add_argument('--output-file', help='Path to output file (default: auto-generated)')
    
    args = parser.parse_args()
    
    # If no log file specified, find the latest
    log_file = args.log_file
    if not log_file:
        log_file = find_latest_log_file()
        if not log_file:
            logger.error("No log files found in results directory")
            return 1
    
    # Extract metrics from log file
    metrics = extract_metrics(log_file)
    if not metrics:
        logger.error("Failed to extract metrics from log file")
        return 1
    
    # Save metrics summary
    output_file = save_metrics_summary(metrics, args.output_file)
    print(f"Metrics summary saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    main()

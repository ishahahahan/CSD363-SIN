# Community Pipeline

A modular Python package that implements a Hybrid Community Detection model for analyzing network community structures, with a focus on the SNAP LiveJournal dataset.

## Overview

This package provides an end-to-end pipeline for detecting and analyzing community structures in large social networks using a hybrid approach combining multiple community detection algorithms:

1. **Louvain Method**: For initial community detection
2. **Girvan-Newman Refinement**: For splitting large communities
3. **Infomap Enhancement**: For improving communities with low modularity

## Installation

### Prerequisites

- Python 3.7 or higher
- NetworkX
- python-louvain (community module)
- matplotlib
- numpy
- scikit-learn
- tqdm
- PyYAML
- igraph (optional, for Infomap enhancement)
- infomap (optional, for enhanced detection)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/community_pipeline.git
cd community_pipeline
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

To install optional dependencies for Infomap enhancement:
```bash
pip install python-igraph
pip install infomap
```

## Usage

### Command Line Interface

The package provides a command-line interface with various configuration options:

```bash
python -m community_pipeline [options]
```

#### Available Options

- `--data-dir`: Directory to store/find data (default: `data`)
- `--sample-size`: Number of edges to sample (default: `None`, use full graph)
- `--size-threshold`: Size threshold for community refinement (default: `1000`)
- `--target-subcommunities`: Target number of subcommunities (default: `5`)
- `--modularity-threshold`: Threshold for Infomap enhancement (default: `0.3`)
- `--config`: Path to config file in YAML or JSON format

### Configuration File

Instead of command-line arguments, you can use a YAML or JSON configuration file:

```yaml
# Example config.yaml
data_dir: data
sample_size: 100000  # Set to null for full graph
size_threshold: 1000
target_subcommunities: 5
modularity_threshold: 0.3
```

Then run with:
```bash
python -m community_pipeline --config config.yaml
```

Note: Command-line arguments override settings in the config file.

## Pipeline Components

### 1. Data Ingestion (`data_io.py`)

- Automatically downloads the LiveJournal dataset
- Extracts and processes the edge list
- Implements caching to avoid redundant processing
- Supports graph sampling for faster experimentation

### 2. Community Detection (`detection.py`)

The detection process follows three phases:

1. **Louvain Phase**: Detects initial communities using the Louvain method for modularity optimization
2. **Refinement Phase**: Uses Girvan-Newman edge betweenness to split large communities
3. **Enhancement Phase**: Applies Infomap to improve communities with low modularity

### 3. Evaluation (`evaluation.py`)

Multiple metrics are computed to evaluate the quality of detected communities:

- **Modularity**: Measures the density of links within communities versus links between communities
- **Conductance**: Measures the fraction of outgoing edges from communities
- **NMI (Normalized Mutual Information)**: When ground truth is available, measures agreement between detected and true communities

### 4. Visualization (`visualization.py`)

Creates visual representations of the detected community structures:

- Handles large graphs by intelligent sampling
- Color-codes nodes by community
- Labels representative nodes for community identification

## Examples

### Basic Run with Default Settings

```bash
# Create data directory
mkdir -p data

# Run the pipeline with defaults
python -m community_pipeline
```

### Run with Sampling for Fast Testing

```bash
python -m community_pipeline --sample-size 50000 --size-threshold 500
```

### Using Custom Configuration

```bash
# Create a custom config file
cat > my_config.yaml << EOF
data_dir: my_data
sample_size: 200000
size_threshold: 2000
target_subcommunities: 8
modularity_threshold: 0.4
EOF

# Run with custom config
python -m community_pipeline --config my_config.yaml
```

## Output

The pipeline produces:

1. **Log Files**: Both console output (INFO level) and detailed log file (`pipeline.log` at DEBUG level)
2. **Visualization**: A PNG image showing the community structure (`community_visualization.png`)
3. **Runtime Statistics**: Performance metrics and timing information

## Running Tests

The package includes unit tests to verify its functionality:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_detection.py
```

## Performance Considerations

- The full LiveJournal dataset is large (~4M nodes, ~35M edges) and may require significant memory
- Use the `--sample-size` option for testing or on machines with limited resources
- The package includes performance safeguards and warnings for operations taking longer than 60 seconds
- Caching is implemented to avoid redundant computations

## Extending the Pipeline

### Adding New Detection Algorithms

1. Implement your algorithm in `detection.py`
2. Update `__main__.py` to include your algorithm in the pipeline
3. Add appropriate parameters to the CLI and configuration options

### Supporting Additional Datasets

To use with other graph datasets:

1. Modify `data_io.py` to handle the new data format
2. Update the download URL and file handling if needed
3. Adjust parameters based on the scale and characteristics of your network

## Troubleshooting

### Common Issues

1. **Memory errors**: Try reducing the sample size or increasing available memory
2. **Missing modules**: Ensure all required packages are installed
3. **Long runtime**: For large datasets, consider sampling or running on more powerful hardware

### Debug Mode

For more detailed logs, check the `pipeline.log` file which contains DEBUG level information.

## License

[MIT License](LICENSE)

## Citation

If using this code for research, please cite:

```
@software{community_pipeline,
  author = {Your Name},
  title = {Community Pipeline: A Hybrid Community Detection Model},
  year = {2025},
  url = {https://github.com/yourusername/community_pipeline}
}
```

## Requirements File

```
# requirements.txt
networkx>=2.5
python-louvain>=0.15
matplotlib>=3.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
tqdm>=4.50.0
pyyaml>=5.4.0
# Optional dependencies
# python-igraph>=0.9.0
# infomap>=1.0.0
```

# Hybrid Community Detection Pipeline

A Python-based, hybrid community detection pipeline that leverages three complementary algorithms—Louvain, Girvan–Newman, and Infomap—to uncover multi-scale community structure in large social graphs (e.g., SNAP LiveJournal dataset). The pipeline then evaluates results via modularity, conductance, coverage, and Normalized Mutual Information (NMI) against ground-truth communities.

## 🚀 Key Features

* **Hybrid Detection**
  * **Louvain** for fast, high-modularity initial clustering.
  * **Girvan–Newman** refinement on large communities using edge betweenness.
  * **Infomap** enhancement for low-modularity clusters via flow-based random walks.
* **Comprehensive Evaluation** : Modularity, average conductance, coverage, and NMI against ground truth.
* **Visualization Suite** : Community size distributions, structural layouts, inter-community links, detailed subgraph views.
* **Configurable & Extensible** : All parameters in `config.yaml` or via CLI flags.

## 📦 Installation & Dependencies

Use a virtual environment and install from `requirements.txt`:

```bash
python3 -m venv venv        # Create virtual environment
source venv/bin/activate     # Activate on Linux/macOS
venv\\Scripts\\activate    # Activate on Windows
pip install -r requirements.txt
```

```
networkx>=2.6
python-louvain>=0.15
infomap>=2.8.0
matplotlib>=3.3
seaborn>=0.11
numpy>=1.19
pandas>=1.1
scikit-learn>=0.24
pyyaml>=5.4
tqdm>=4.60
```

## 📊 Dataset

We use the SNAP LiveJournal network and its user-defined groups as ground-truth:

* **Edge list** : `com-lj.ungraph.txt.gz` (undirected graph)
* **Ground truth** : `com-lj.all.cmty.txt.gz` (all communities) or `com-lj.top5000.cmty.txt.gz` for NMI evaluation.

## ⚡ Quick Start

1. **Configure** `config.yaml` (paths, thresholds, performance flags).
2. **Run pipeline** (downloads data if needed):
   ```bash
   python __main__.py --config=config.yaml
   ```
3. **Inspect outputs** :

* `results/metrics_*.json` & `results/metrics_summary_*.txt` (evaluation)
* `community_visualizations/` (PNG & HTML reports)

## ⚙️ Configuration (`config.yaml`)

```yaml
data_dir: data
sample_size: null          # full graph (~4.8M nodes, ~34.7M edges)
edge_file: com-lj.ungraph.txt.gz
ground_truth_file: com-lj.all.cmty.txt.gz

# Community detection parameters
size_threshold: 1000       # refine communities >1k nodes
target_subcommunities: 5
modularity_threshold: 0.3

# Performance tuning
max_iterations: 30
time_limit: 36000          # seconds per stage
fast_mode: true

# Output directories
output_dir: results
visualization_dir: community_visualizations
```

## 🎛️ CLI Reference

```text
python __main__.py [--config CONFIG] [--data-dir DIR] [--sample-size N]
                   [--size-threshold N] [--target-subcommunities N]
                   [--modularity-threshold F] [--max-iterations N]
                   [--time-limit S] [--fast-mode]
                   [--input-edge-file FILE] [--ground-truth-file FILE]
```

## 📈 Evaluation Metrics

* **Modularity** : Quality of partitioning.
* **Conductance** : Edge-boundary measure per community.
* **Coverage** : Fraction of intra-community edges.
* **NMI** : Agreement with ground truth (0–1 scale).

Compare metrics before/after each stage to quantify improvements.

## 📂 Project Structure

```
├── algorithms/            # Wrappers for GN & Infomap
├── data/                  # Raw & extracted SNAP files
├── evaluation/            # Metrics computation & summary scripts
├── visualization/         # Plotting & HTML report generation
├── __main__.py            # Pipeline entry point
├── config.yaml            # Default configuration
├── requirements.txt       # Python dependencies
├── extract_metrics.py     # Log → summary TXT
└── README.md              # This file
```
